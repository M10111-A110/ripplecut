"""Load the canonical model configuration (JSON) into frozen schema objects.

Configuration is the single source of truth for services, topology, dependency
rules, criticality, actions, costs, conflicts, effects and state thresholds
(master spec §54, §97). Parsing is strict; semantic consistency checks live in
``ripplecut.model.validation`` and run before anything is returned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..errors import ModelValidationError
from .schema import (DOWN, UP, ActionConflict, ActionUniverse, ConflictResolution, DependencyGroup,
                     DependencyInput, DependencyRule, Effect, Intervention, ObjectivePolicy, Precondition,
                     RuleType, Service, ServiceRole, Strength, SystemModel, frozen_mapping)
from .validation import validate_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "config" / "online_boutique_canonical.json"


@dataclass(frozen=True)
class ModelBundle:
    system: SystemModel
    actions: ActionUniverse
    objective: ObjectivePolicy
    state_estimation: Mapping[str, Any]
    presentation: Mapping[str, Any]
    source_path: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=frozen_mapping)


class _Issues:
    def __init__(self) -> None:
        self.items: List[str] = []

    def add(self, msg: str) -> None:
        self.items.append(msg)

    def raise_if_any(self, context: str) -> None:
        if self.items:
            raise ModelValidationError(f"{context}: {len(self.items)} issue(s): " + "; ".join(self.items),
                                       details={"issues": list(self.items)})


def _req(d: Mapping[str, Any], key: str, where: str, issues: _Issues, default: Any = None) -> Any:
    if key not in d:
        issues.add(f"{where}: missing required field '{key}'")
        return default
    return d[key]


def _parse_cost(value: Any, where: str, issues: _Issues) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        issues.add(f"{where}: cost must be a number, got {value!r}")
        return Fraction(0)
    try:
        cost = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        issues.add(f"{where}: cost {value!r} is not a number")
        return Fraction(0)
    if cost < 0:
        issues.add(f"{where}: cost must be nonnegative (got {value})")
    return cost


def _parse_effect(d: Any, where: str, issues: _Issues) -> Effect:
    if not isinstance(d, Mapping):
        issues.add(f"{where}: effect must be an object with set_up/set_down")
        return Effect()
    unknown = set(d) - {"set_up", "set_down"}
    if unknown:
        issues.add(f"{where}: unknown effect keys {sorted(unknown)} (only set_up/set_down are modeled)")
    up = d.get("set_up", [])
    down = d.get("set_down", [])
    if not isinstance(up, list) or not isinstance(down, list):
        issues.add(f"{where}: set_up/set_down must be lists")
        return Effect()
    return Effect(set_up=frozenset(up), set_down=frozenset(down))


def _parse_state_word(word: Any, where: str, issues: _Issues) -> int:
    if word == "UP":
        return UP
    if word == "DOWN":
        return DOWN
    issues.add(f"{where}: state must be 'UP' or 'DOWN' in the formal Boolean model (got {word!r})")
    return UP


def parse_model(raw: Mapping[str, Any], source_path: Optional[str] = None) -> ModelBundle:
    issues = _Issues()

    # ---- services -------------------------------------------------------
    services: List[Service] = []
    for i, s in enumerate(_req(raw, "services", "model", issues, [])):
        where = f"services[{i}]"
        sid = _req(s, "id", where, issues, f"<missing-{i}>")
        role_word = s.get("role", "business")
        try:
            role = ServiceRole(role_word)
        except ValueError:
            issues.add(f"{where}: unknown role {role_word!r}")
            role = ServiceRole.BUSINESS
        services.append(Service(id=sid, name=s.get("name", sid), role=role,
                                description=s.get("description", ""),
                                aliases=tuple(s.get("aliases", [])), source=s.get("source", "")))
    services.sort(key=lambda s: s.id)

    # ---- dependency rules -----------------------------------------------
    rules: Dict[str, DependencyRule] = {}
    for i, r in enumerate(raw.get("dependency_rules", [])):
        where = f"dependency_rules[{i}]"
        downstream = _req(r, "downstream", where, issues, f"<missing-{i}>")
        rt_word = _req(r, "rule_type", where, issues, "AND")
        try:
            rule_type = RuleType(rt_word)
        except ValueError:
            issues.add(f"{where}: unknown rule_type {rt_word!r}")
            rule_type = RuleType.AND
        inputs = []
        for j, inp in enumerate(r.get("inputs", [])):
            w2 = f"{where}.inputs[{j}]"
            try:
                strength = Strength(inp.get("strength", "hard"))
            except ValueError:
                issues.add(f"{w2}: strength must be 'hard' or 'soft'")
                strength = Strength.HARD
            inputs.append(DependencyInput(service=_req(inp, "service", w2, issues, ""), strength=strength,
                                          topology=inp.get("topology", "UNKNOWN — REQUIRES VERIFICATION"),
                                          evidence=inp.get("evidence", "")))
        groups = tuple(DependencyGroup(members=tuple(g.get("members", [])), q=g.get("q", 0))
                       for g in r.get("groups", []))
        if downstream in rules:
            issues.add(f"{where}: duplicate rule for downstream '{downstream}' (one rule per service)")
        rules[downstream] = DependencyRule(downstream=downstream, rule_type=rule_type, inputs=tuple(inputs),
                                           threshold=r.get("threshold"), groups=groups,
                                           semantics=r.get("semantics", "RIPPLECUT-MODELED"),
                                           source=r.get("source", ""), notes=r.get("notes", ""))

    # ---- criticality ------------------------------------------------------
    crit = _req(raw, "criticality", "model", issues, {})
    critical = frozenset(crit.get("critical", []))
    non_critical = frozenset(crit.get("non_critical", []))
    infrastructure = frozenset(crit.get("infrastructure", []))

    # ---- interventions and conflicts -------------------------------------
    conflicts: Dict[frozenset, ActionConflict] = {}
    for i, c in enumerate(raw.get("action_conflicts", [])):
        where = f"action_conflicts[{i}]"
        pair = c.get("actions", [])
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
            issues.add(f"{where}: 'actions' must list exactly two distinct action ids")
            continue
        key = frozenset(pair)
        try:
            resolution = ConflictResolution(c.get("resolution"))
        except ValueError:
            issues.add(f"{where}: resolution must be INVALID_COMBINATION or DEFINED_JOINT_EFFECT")
            continue
        joint = None
        if resolution is ConflictResolution.DEFINED_JOINT_EFFECT:
            if "joint_effect" not in c:
                issues.add(f"{where}: DEFINED_JOINT_EFFECT requires an explicit 'joint_effect'")
            joint = _parse_effect(c.get("joint_effect", {}), where + ".joint_effect", issues)
        elif "joint_effect" in c:
            issues.add(f"{where}: INVALID_COMBINATION must not define a joint_effect")
        if key in conflicts:
            issues.add(f"{where}: duplicate conflict declaration for {sorted(key)}")
        conflicts[key] = ActionConflict(actions=key, resolution=resolution, joint_effect=joint,
                                        rationale=c.get("rationale", ""))

    interventions: List[Intervention] = []
    for i, a in enumerate(raw.get("interventions", [])):
        where = f"interventions[{i}]"
        aid = _req(a, "id", where, issues, f"<missing-{i}>")
        pre = tuple(Precondition(service=p.get("service", ""),
                                 state=_parse_state_word(p.get("state"), f"{where}.preconditions", issues))
                    for p in a.get("preconditions", []))
        conflicts_for = tuple(sorted(next(iter(k - {aid})) for k in conflicts if aid in k))
        interventions.append(Intervention(
            id=aid, name=a.get("name", aid), description=a.get("description", ""),
            cost=_parse_cost(_req(a, "cost", where, issues, 0), where, issues),
            targets=tuple(a.get("targets", [])), preconditions=pre,
            effect=_parse_effect(_req(a, "effect", where, issues, {}), where + ".effect", issues),
            conflicts=conflicts_for, metadata=frozen_mapping(a.get("metadata", {}))))
    interventions.sort(key=lambda a: a.id)

    # ---- objective policy ---------------------------------------------------
    obj_raw = raw.get("objective_policy", {})
    objective = ObjectivePolicy()
    if obj_raw:
        if obj_raw.get("type", "LEXICOGRAPHIC") != "LEXICOGRAPHIC":
            issues.add("objective_policy.type: only LEXICOGRAPHIC is part of the frozen MVP "
                       "(weighted scores are a future extension)")
        if tuple(obj_raw.get("order", objective.order)) != objective.order:
            issues.add(f"objective_policy.order must be exactly {list(objective.order)} (frozen, PDF §8.3)")
        if obj_raw.get("final_tie_breaker", objective.final_tie_breaker) != objective.final_tie_breaker:
            issues.add("objective_policy.final_tie_breaker must be 'sorted_action_ids'")

    issues.raise_if_any("model configuration could not be parsed")

    meta = raw.get("model", {})
    system = SystemModel(model_id=meta.get("id", "unnamed"), name=meta.get("name", "unnamed"),
                         services=tuple(services), rules=frozen_mapping(rules), critical=critical,
                         non_critical=non_critical, infrastructure=infrastructure,
                         metadata=frozen_mapping(meta))
    actions = ActionUniverse(interventions=tuple(interventions), conflicts=frozen_mapping(conflicts))
    bundle = ModelBundle(system=system, actions=actions, objective=objective,
                         state_estimation=frozen_mapping(raw.get("state_estimation", {})),
                         presentation=frozen_mapping(raw.get("presentation", {})),
                         source_path=source_path, raw=frozen_mapping(raw))
    validate_model(bundle.system, bundle.actions)   # raises ModelValidationError on any inconsistency
    return bundle


def load_model(path: Optional[Path] = None) -> ModelBundle:
    p = Path(path) if path else DEFAULT_MODEL_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelValidationError(f"model file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ModelValidationError(f"model file is not valid JSON: {p}: {exc}") from exc
    return parse_model(raw, source_path=str(p))
