"""Canonical structured incident (master spec §6.1, §7; PDF §14.1-§14.2).

The structured representation is canonical; natural language is optional and
must be converted into this schema and validated before anything reaches the
deterministic engine. Validation is strict: unknown keys, unknown services and
malformed values are rejected, never repaired by guessing.

Formal mapping (PDF §4.1): x_v = 1 iff v is UP. A service reported DEGRADED is
kept as ``degraded_services`` for display but is 0 in the formal state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from ..errors import ErrorCode, IncidentError
from ..model.schema import State, SystemModel

STRUCTURED_KEYS = frozenset({"failed_services", "degraded_services", "scenario_id"})
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


@dataclass(frozen=True)
class StructuredIncident:
    failed_services: Tuple[str, ...]
    degraded_services: Tuple[str, ...] = ()
    scenario_id: Optional[str] = None
    source: str = "structured"             # structured | rule_based_parser | llm_parser | replay_fixture
    raw_text: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def not_up(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.failed_services) | set(self.degraded_services)))

    def initial_state(self, system: SystemModel) -> State:
        """x(0): every reported DOWN or DEGRADED service is 0, every other service is 1."""
        return system.state_with_failures(self.not_up)

    def canonical(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"failed_services": list(self.failed_services)}
        if self.degraded_services:
            out["degraded_services"] = list(self.degraded_services)
        if self.scenario_id:
            out["scenario_id"] = self.scenario_id
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {**self.canonical(), "source": self.source, "raw_text": self.raw_text, "notes": list(self.notes),
                "formal_initial_down": list(self.not_up)}


def _service_list(value: Any, key: str, system: SystemModel, code_on_type: ErrorCode) -> Tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise IncidentError(f"'{key}' must be a list of service ids", code_on_type, {"field": key})
    unknown = sorted({v for v in value if not system.has_service(v)})
    if unknown:
        raise IncidentError(f"unknown service id(s) in '{key}': {unknown}; allowed: {list(system.service_ids)}",
                            ErrorCode.INVALID_SERVICE, {"field": key, "unknown": unknown})
    return tuple(sorted(set(value)))


def validate_incident(raw: Any, system: SystemModel, *, source: str = "structured",
                      raw_text: Optional[str] = None, extra_allowed: Iterable[str] = (),
                      notes: Iterable[str] = ()) -> StructuredIncident:
    """Strictly validate a structured incident against the system model."""
    code = ErrorCode.LLM_PARSE_ERROR if source == "llm_parser" else ErrorCode.CONFIG_ERROR
    if not isinstance(raw, Mapping):
        raise IncidentError("structured incident must be a JSON object", code)
    allowed: FrozenSet[str] = STRUCTURED_KEYS | frozenset(extra_allowed)
    unknown_keys = sorted(set(raw) - allowed)
    if unknown_keys:
        raise IncidentError(f"structured incident has unsupported field(s) {unknown_keys}; only "
                            f"{sorted(STRUCTURED_KEYS)} may enter the deterministic engine "
                            f"(actions, dependencies, costs and outcomes are never accepted from input)",
                            code, {"unknown_fields": unknown_keys})
    if "failed_services" not in raw:
        raise IncidentError("structured incident needs 'failed_services' (may be an empty list)", code)
    failed = _service_list(raw["failed_services"], "failed_services", system, code)
    degraded = _service_list(raw.get("degraded_services", []), "degraded_services", system, code)
    both = sorted(set(failed) & set(degraded))
    if both:
        raise IncidentError(f"service(s) {both} are listed as both failed and degraded", ErrorCode.INCIDENT_AMBIGUOUS,
                            {"services": both})
    scenario_id = raw.get("scenario_id")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not _SCENARIO_ID.match(scenario_id)):
        raise IncidentError("scenario_id must match [A-Za-z0-9_.-]{1,64}", code)
    return StructuredIncident(failed_services=failed, degraded_services=degraded, scenario_id=scenario_id,
                              source=source, raw_text=raw_text, notes=tuple(notes))


def hard_ancestors(system: SystemModel, service: str) -> FrozenSet[str]:
    """All services reachable from ``service`` by following hard dependency edges upstream."""
    seen: set = set()
    stack: List[str] = [service]
    while stack:
        s = stack.pop()
        rule = system.rules.get(s)
        if rule is None:
            continue
        for u in rule.hard_upstream:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    seen.discard(service)
    return frozenset(seen)


def possible_consequences(system: SystemModel, reported: Iterable[str]) -> Dict[str, List[str]]:
    """Reported services that could be *consequences* of another reported failure under the model.

    A natural-language report cannot tell whether such a service failed
    independently or is a symptom of the cascade; the parsers therefore return
    an explicit ambiguity instead of guessing (master spec §7).
    """
    rep = set(reported)
    out: Dict[str, List[str]] = {}
    for s in sorted(rep):
        causes = sorted(hard_ancestors(system, s) & (rep - {s}))
        if causes:
            out[s] = causes
    return out
