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

PAST_TEMPORAL_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "five minutes ago", "10 minutes ago", "15 minutes ago", "earlier today", "earlier", "previously",
        "yesterday", "in the past", "last night", "last hour", "was down earlier", "failed earlier",
        "was down", "was failing", "was degraded", "was slow", "was broken", "was offline",
        "was unreachable", "had failed", "had crashed", "used to be", "prior"
    ]
], key=lambda item: -len(item)))

RECOVERY_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "healthy now", "is healthy now", "up now", "is up now", "recovered", "back up", "is back up",
        "fixed now", "is fixed now", "resolved now", "is resolved now", "fine now", "is fine now",
        "operational now", "is operational now", "working now", "is working now", "normal now",
        "is normal now", "all good now", "restored", "is restored", "it is recovered", "recovered now"
    ]
], key=lambda item: -len(item)))

UNCERTAINTY_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "might be", "could be", "possibly", "maybe", "suspected", "suspecting",
        "potential", "potentially", "unconfirmed", "seems to be", "appears to be",
        "perhaps", "might have", "not sure if", "uncertain"
    ]
], key=lambda item: -len(item)))

RETRACTION_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "just kidding it is up", "just kidding it's up", "just kidding", "never mind",
        "nevermind", "ignore that", "scratch that", "false alarm", "disregard", "cancel that"
    ]
], key=lambda item: -len(item)))

HYPOTHETICAL_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "what if", "suppose", "assuming", "hypothetically", "imagine", "what happens if", "in case of"
    ]
], key=lambda item: -len(item)))

MAINTENANCE_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(sorted([
    tuple(p.split()) for p in [
        "scheduled maintenance", "planned maintenance", "maintenance window", "under maintenance",
        "routine maintenance", "scheduled downtime", "planned downtime"
    ]
], key=lambda item: -len(item)))

# Components that do not exist in the Online-Boutique model; naming one makes the
# incident unrepresentable (PDF §1.2: generic example components are not OB services).
UNMODELED_TERMS: Tuple[Tuple[str, ...], ...] = tuple(tuple(t.split()) for t in [
    "database", "db", "auth service", "authentication", "api gateway", "inventory", "dns", "cdn",
    "load balancer", "kafka"])

_CLAUSE_SPLIT = re.compile(r"[.;!?\n]+|\b(?:but|however|although|though|whereas|while)\b")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_SERVICE_LIKE = re.compile(r"^[a-z][a-z0-9-]*service$")


def _match_phrase_tuples(toks: Sequence[str], phrases: Sequence[Tuple[str, ...]]) -> List[Tuple[int, int, str]]:
    """Find non-overlapping matches of multi-word phrase tuples in tokens, longest first.
    Returns sorted list of (start_idx, end_idx, matched_phrase_str)."""
    matches = []
    used = [False] * len(toks)
    for phrase in phrases:
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            if not any(used[i:i + n]) and tuple(toks[i:i + n]) == phrase:
                for j in range(i, i + n):
                    used[j] = True
                matches.append((i, i + n, " ".join(phrase)))
    return sorted(matches)


@dataclass
class Mention:
    service: str
    alias: str
    clause: int
    position: int
    state: Optional[str] = None
    state_phrase: Optional[str] = None
    temporal: str = "current"        # current | historical | recovery | uncertain | hypothetical | maintenance
    confidence: str = "HIGH"         # HIGH | UNCERTAIN | AMBIGUOUS
    source_text: str = ""
    issue: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"service": self.service, "matched_text": self.alias, "clause": self.clause,
                "state": self.state, "state_phrase": self.state_phrase, "temporal": self.temporal,
                "confidence": self.confidence, "source_text": self.source_text, "issue": self.issue}


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
        clauses = split_clauses(text)
        prev_clause_mentions: List[Mention] = []

        # Text-level global checks
        text_toks = tokenize(text)
        global_hypotheticals = _match_phrase_tuples(text_toks, HYPOTHETICAL_PHRASES)
        global_retractions = _match_phrase_tuples(text_toks, RETRACTION_PHRASES)

        for ci, clause in enumerate(clauses):
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
                        m = Mention(sid, " ".join(alias), ci, i, source_text=clause)
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

            # 3) qualifiers in clause
            hypotheticals = _match_phrase_tuples(toks, HYPOTHETICAL_PHRASES) or (global_hypotheticals if ci == 0 else [])
            maintenances = _match_phrase_tuples(toks, MAINTENANCE_PHRASES)
            retractions = _match_phrase_tuples(toks, RETRACTION_PHRASES) or global_retractions
            uncertainties = _match_phrase_tuples(toks, UNCERTAINTY_PHRASES)
            pasts = _match_phrase_tuples(toks, PAST_TEMPORAL_PHRASES)
            recoveries = _match_phrase_tuples(toks, RECOVERY_PHRASES)
            statuses = self._statuses(toks, used)

            # If this clause has no service mentions but has recovery/retraction/status,
            # and previous clause had mentions, apply clause info to previous mentions
            if not clause_mentions and prev_clause_mentions:
                if retractions:
                    for pm in prev_clause_mentions:
                        pm.confidence = "AMBIGUOUS"
                        pm.issue = pm.issue or f"retracted statement ('{retractions[0][2]}') - state is ambiguous"
                elif recoveries:
                    for pm in prev_clause_mentions:
                        pm.state = UP_WORD
                        pm.state_phrase = recoveries[0][2]
                        pm.temporal = "recovery"
                        pm.confidence = "HIGH"
                        pm.issue = None
                elif statuses:
                    for pm in prev_clause_mentions:
                        self._assign(pm, statuses, uncertainties, pasts, recoveries, hypotheticals, maintenances, retractions)

            # 4) status phrases and their assignment to mentions
            for m in sorted(clause_mentions, key=lambda m: m.position):
                self._assign(m, statuses, uncertainties, pasts, recoveries, hypotheticals, maintenances, retractions,
                             all_clause_mentions=clause_mentions)

            if clause_mentions:
                prev_clause_mentions = clause_mentions
                mentions.extend(sorted(clause_mentions, key=lambda m: m.position))

        if global_hypotheticals:
            for m in mentions:
                m.temporal = "hypothetical"
                m.issue = m.issue or f"hypothetical scenario ('{global_hypotheticals[0][2]}') - not an active incident"

        if global_retractions:
            for m in mentions:
                m.confidence = "AMBIGUOUS"
                m.issue = m.issue or f"retracted statement ('{global_retractions[0][2]}') - state is ambiguous"

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

    @classmethod
    def _assign(cls, m: Mention, statuses: List[Tuple[int, str, str, bool]],
                uncertainties: List[Tuple[int, int, str]],
                pasts: List[Tuple[int, int, str]],
                recoveries: List[Tuple[int, int, str]],
                hypotheticals: List[Tuple[int, int, str]],
                maintenances: List[Tuple[int, int, str]],
                retractions: List[Tuple[int, int, str]],
                all_clause_mentions: Optional[List[Mention]] = None) -> None:
        if hypotheticals:
            m.temporal = "hypothetical"
            m.issue = m.issue or f"hypothetical scenario ('{hypotheticals[0][2]}') - not an active incident"
            return

        if maintenances:
            m.temporal = "maintenance"
            m.issue = m.issue or f"maintenance window ('{maintenances[0][2]}') - not an unexpected outage"
            return

        if retractions:
            m.confidence = "AMBIGUOUS"
            m.issue = m.issue or f"retracted statement ('{retractions[0][2]}') - state is ambiguous"
            return

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

        # Check if an uncertainty phrase qualifies this status
        m_uncert = [u for u in uncertainties if abs(u[0] - m.position) <= 5 or (min(m.position, chosen[0]) - 3 <= u[0] <= max(m.position, chosen[0]) + 2)]
        if m_uncert:
            m.confidence = "UNCERTAIN"
            m.issue = m.issue or f"uncertain state ('{m_uncert[0][2]}') - RippleCut requires confirmed state"
            m.state_phrase = phrase
            return

        # Check if a recovery phrase qualifies this mention
        m_rec = [r for r in recoveries if abs(r[0] - m.position) <= 6 or abs(r[0] - chosen[0]) <= 5]
        if m_rec:
            m.temporal = "recovery"
            m.state = UP_WORD
            m.state_phrase = m_rec[0][2]
            m.confidence = "HIGH"
            m.issue = None
            return

        # Check if a past temporal phrase qualifies this status
        m_past = [p for p in pasts if (min(m.position, chosen[0]) - 1 <= p[0] <= max(m.position, chosen[0]) + 4)]
        if m_past:
            m.temporal = "historical"
            m.issue = m.issue or f"historical report without current confirmation ('{m_past[0][2]}') - state is ambiguous"
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
