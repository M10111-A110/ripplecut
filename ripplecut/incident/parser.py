"""Deterministic natural-language incident parser (offline default; master spec §104-§105).

This parser is the offline fallback for the optional LLM parser and follows the
same contract: it extracts *candidate* service mentions and states, and it
returns an explicit non-OK status instead of guessing whenever the text is not
fully representable in the model:

  OK                  every mentioned service is known and has an unambiguous state
  AMBIGUOUS           a mention has no/contradictory/negated state, or a reported
                      failure could be a consequence of another reported failure
                      (alternatives are offered; the user must choose)
  UNKNOWN_SERVICE     the text names a component that is not in the model
  NO_SERVICE_FOUND    no known service is mentioned

Service vocabulary (ids, names, aliases, excluded components) comes from the
model configuration. Only the status vocabulary below lives in code; it is a
parser concern, not a model fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..errors import IncidentError
from ..model.schema import SystemModel
from .schema import StructuredIncident, possible_consequences, validate_incident

OK = "OK"
AMBIGUOUS = "AMBIGUOUS"
UNKNOWN_SERVICE = "UNKNOWN_SERVICE"
NO_SERVICE_FOUND = "NO_SERVICE_FOUND"
LLM_PARSE_ERROR = "LLM_PARSE_ERROR"

DOWN, DEGRADED, UP_WORD = "DOWN", "DEGRADED", "UP"

# (phrase tokens, state). Multi-word phrases are matched first.
STATUS_PHRASES: Tuple[Tuple[Tuple[str, ...], str], ...] = tuple(sorted([
    *((tuple(p.split()), DOWN) for p in [
        "down", "failed", "failing", "fail", "fails", "failure", "failures", "crashed", "crashing", "crash",
        "crashes", "unavailable", "outage", "dead", "unreachable", "offline", "broken", "erroring", "errors",
        "unresponsive", "500", "5xx", "503", "not working", "not responding", "not reachable", "went down",
        "out of service"]),
    *((tuple(p.split()), DEGRADED) for p in [
        "slow", "slower", "degraded", "degrading", "latency", "lagging", "sluggish", "timing out", "timeouts",
        "flaky", "intermittent", "intermittently"]),
    *((tuple(p.split()), UP_WORD) for p in [
        "up", "healthy", "fine", "ok", "okay", "operational", "recovered", "normal", "working", "available"]),
], key=lambda item: -len(item[0])))

NEGATIONS = frozenset({"not", "no", "never", "without", "longer"})

# Components that do not exist in the Online-Boutique model; naming one makes the
# incident unrepresentable (PDF §1.2: generic example components are not OB services).
UNMODELED_TERMS: Tuple[Tuple[str, ...], ...] = tuple(tuple(t.split()) for t in [
    "database", "db", "auth service", "authentication", "api gateway", "inventory", "dns", "cdn",
    "load balancer", "kafka"])

_CLAUSE_SPLIT = re.compile(r"[.;!?\n]+|\b(?:but|however|although|though|whereas|while)\b")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_SERVICE_LIKE = re.compile(r"^[a-z][a-z0-9-]*service$")


@dataclass
class Mention:
    service: str
    alias: str
    clause: int
    position: int
    state: Optional[str] = None
    state_phrase: Optional[str] = None
    issue: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"service": self.service, "matched_text": self.alias, "clause": self.clause,
                "state": self.state, "state_phrase": self.state_phrase, "issue": self.issue}


@dataclass
class ParseResult:
    status: str
    parser: str
    text: str
    incident: Optional[StructuredIncident] = None
    mentions: List[Mention] = field(default_factory=list)
    message: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    unknown_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "parser": self.parser, "text": self.text,
                "structured_incident": self.incident.to_dict() if self.incident else None,
                "mentions": [m.to_dict() for m in self.mentions], "message": self.message,
                "alternatives": self.alternatives, "unknown_terms": self.unknown_terms, "notes": self.notes}


def tokenize(text: str) -> List[str]:
    t = text.lower().replace("\u2019", "'").replace("n't", " not")
    return _TOKEN.findall(t)


def split_clauses(text: str) -> List[str]:
    t = text.lower().replace("\u2019", "'").replace("n't", " not")
    return [c for c in _CLAUSE_SPLIT.split(t) if c and c.strip()]


def _alias_table(system: SystemModel, raw_model: Optional[Mapping[str, Any]]) -> Tuple[
        List[Tuple[Tuple[str, ...], str]], List[Tuple[Tuple[str, ...], str]], Dict[Tuple[str, ...], List[str]]]:
    known: Dict[Tuple[str, ...], set] = {}
    for s in system.services:
        for a in {s.id, s.name, *s.aliases}:
            toks = tuple(tokenize(a))
            if toks:
                known.setdefault(toks, set()).add(s.id)
    collisions = {k: sorted(v) for k, v in known.items() if len(v) > 1}
    table = sorted(((k, sorted(v)[0]) for k, v in known.items()), key=lambda kv: (-len(kv[0]), kv[0]))
    excluded: List[Tuple[Tuple[str, ...], str]] = []
    for e in (raw_model or {}).get("excluded_services", []) or []:
        for a in {e.get("id", ""), *e.get("aliases", [])}:
            toks = tuple(tokenize(a))
            if toks:
                excluded.append((toks, e.get("id", a)))
    excluded += [(t, " ".join(t)) for t in UNMODELED_TERMS]
    excluded.sort(key=lambda kv: -len(kv[0]))
    return table, excluded, collisions


class RuleBasedParser:
    name = "rule_based_parser"

    def __init__(self, system: SystemModel, raw_model: Optional[Mapping[str, Any]] = None) -> None:
        self.system = system
        self.table, self.excluded, self.collisions = _alias_table(system, raw_model)
        self.known_tokens = {t for toks, _ in self.table for t in toks}

    # ------------------------------------------------------------------------
    def parse(self, text: str) -> ParseResult:
        if not isinstance(text, str) or not text.strip():
            return ParseResult(NO_SERVICE_FOUND, self.name, str(text), message="empty incident text")
        mentions: List[Mention] = []
        unknown: List[str] = []
        for ci, clause in enumerate(split_clauses(text)):
            toks = tokenize(clause)
            used = [False] * len(toks)
            clause_mentions: List[Mention] = []
            # 1) known services (longest alias first, non-overlapping)
            for alias, sid in self.table:
                n = len(alias)
                for i in range(len(toks) - n + 1):
                    if not any(used[i:i + n]) and tuple(toks[i:i + n]) == alias:
                        for j in range(i, i + n):
                            used[j] = True
                        m = Mention(sid, " ".join(alias), ci, i)
                        if alias in self.collisions:
                            m.issue = f"'{' '.join(alias)}' matches several services {self.collisions[alias]}"
                        clause_mentions.append(m)
            # 2) components that are not part of the model
            for alias, label in self.excluded:
                n = len(alias)
                for i in range(len(toks) - n + 1):
                    if not any(used[i:i + n]) and tuple(toks[i:i + n]) == alias:
                        for j in range(i, i + n):
                            used[j] = True
                        unknown.append(label)
            for i, t in enumerate(toks):
                if not used[i] and _SERVICE_LIKE.match(t):
                    used[i] = True
                    unknown.append(t)
            # 3) status phrases and their assignment to mentions
            statuses = self._statuses(toks, used)
            for m in sorted(clause_mentions, key=lambda m: m.position):
                self._assign(m, statuses)
            mentions.extend(sorted(clause_mentions, key=lambda m: m.position))

        result = ParseResult(OK, self.name, text, mentions=mentions)
        if unknown:
            result.status = UNKNOWN_SERVICE
            result.unknown_terms = sorted(set(unknown))
            result.message = (f"the incident mentions component(s) not in the model: {result.unknown_terms}. "
                              f"RippleCut does not guess a mapping; use a modeled service id: "
                              f"{list(self.system.service_ids)}")
            return result
        if not mentions:
            result.status = NO_SERVICE_FOUND
            result.message = "no modeled service is mentioned; provide a structured incident or name a service"
            return result
        return self._finish(result)

    # ------------------------------------------------------------------------
    @staticmethod
    def _statuses(toks: Sequence[str], used: List[bool]) -> List[Tuple[int, str, str, bool]]:
        """(position, state, phrase, negated) for every status phrase in the clause."""
        out = []
        taken = list(used)
        for phrase, state in STATUS_PHRASES:
            n = len(phrase)
            for i in range(len(toks) - n + 1):
                if not any(taken[i:i + n]) and tuple(toks[i:i + n]) == phrase:
                    for j in range(i, i + n):
                        taken[j] = True
                    negated = any(t in NEGATIONS for t in toks[max(0, i - 2):i])
                    out.append((i, state, " ".join(phrase), negated))
        return sorted(out)

    @staticmethod
    def _assign(m: Mention, statuses: List[Tuple[int, str, str, bool]]) -> None:
        after = [s for s in statuses if s[0] > m.position]
        before = [s for s in statuses if s[0] < m.position]
        chosen = after[0] if after else (before[-1] if before else None)
        if chosen is None:
            m.issue = m.issue or "no state is stated for this service"
            return
        _, state, phrase, negated = chosen
        if negated:
            m.issue = m.issue or f"negated state phrase '{phrase}' - state is not stated positively"
            m.state_phrase = phrase
            return
        m.state, m.state_phrase = state, phrase

    def _finish(self, result: ParseResult) -> ParseResult:
        by_service: Dict[str, set] = {}
        for m in result.mentions:
            if m.state:
                by_service.setdefault(m.service, set()).add(m.state)
        issues = [f"{m.service}: {m.issue}" for m in result.mentions if m.issue]
        for sid, states in sorted(by_service.items()):
            if len(states) > 1:
                issues.append(f"{sid}: contradictory states {sorted(states)}")
        if issues:
            result.status = AMBIGUOUS
            result.message = "the incident text is ambiguous; RippleCut does not guess: " + "; ".join(issues)
            return result
        failed = sorted(s for s, st in by_service.items() if st == {DOWN})
        degraded = sorted(s for s, st in by_service.items() if st == {DEGRADED})
        healthy = sorted(s for s, st in by_service.items() if st == {UP_WORD})
        if healthy:
            result.notes.append(f"reported healthy (no change to x(0)): {healthy}")
        return finalize_candidate(self.system, result, failed, degraded, self.name)


def finalize_candidate(system: SystemModel, result: ParseResult, failed: List[str], degraded: List[str],
                       source: str) -> ParseResult:
    """Shared by the rule-based and LLM parsers: schema-validate, then check root/symptom ambiguity."""
    try:
        incident = validate_incident({"failed_services": failed, "degraded_services": degraded}, system,
                                     source=source, raw_text=result.text)
    except IncidentError as exc:
        result.status = AMBIGUOUS if exc.code.value == "INCIDENT_AMBIGUOUS" else LLM_PARSE_ERROR
        result.message = exc.message
        return result
    if not incident.not_up:
        result.status = OK
        result.incident = incident
        result.message = "no service is reported DOWN or DEGRADED"
        return result
    consequences = possible_consequences(system, incident.not_up)
    if consequences:
        roots_failed = [s for s in failed if s not in consequences]
        roots_degraded = [s for s in degraded if s not in consequences]
        desc = "; ".join(f"{s} has hard upstream {c}" for s, c in consequences.items())
        result.status = AMBIGUOUS
        result.message = ("reported failure(s) could be consequences of another reported failure under the model "
                          f"({desc}). The text does not say whether they failed independently; choose one:")
        result.alternatives = [
            {"id": "consequence", "label": f"Treat {sorted(consequences)} as cascade consequences "
                                           f"(initial failures: {sorted(roots_failed + roots_degraded)})",
             "incident": {"failed_services": roots_failed,
                          **({"degraded_services": roots_degraded} if roots_degraded else {})}},
            {"id": "independent", "label": f"Treat all reported services as independent initial failures "
                                           f"({sorted(failed + degraded)})",
             "incident": {"failed_services": failed, **({"degraded_services": degraded} if degraded else {})}},
        ]
        return result
    result.status = OK
    result.incident = incident
    result.message = "parsed; every mentioned service is modeled and has an unambiguous state"
    return result
