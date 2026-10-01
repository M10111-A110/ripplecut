"""Semantic validation of a parsed model (master spec §77 GATE 1, §102).

Fail early: every inconsistency is collected and reported together as a
``MODEL_VALIDATION_ERROR``. Nothing downstream runs on an invalid model.

Besides reference checks, this module enforces the composition rule of the
intervention model: two actions whose effects could contradict each other
(one sets a service UP, the other sets it DOWN) MUST carry an explicit
conflict declaration (INVALID_COMBINATION or DEFINED_JOINT_EFFECT). With these
checks in place, every legal plan has a well-defined joint transformation T_B,
and every illegality reason is *upward closed* (if B is illegal, every
superset of B is illegal) - a property the Branch & Bound solver relies on.
"""
from __future__ import annotations

from itertools import combinations
from typing import List

from ..errors import ModelValidationError
from .schema import (RIPPLECUT_MODELED, SOURCE_BACKED, ActionUniverse, ConflictResolution, Effect,
                     RuleType, ServiceRole, Strength, SystemModel)

ALLOWED_TOPOLOGY_LABELS = {SOURCE_BACKED, RIPPLECUT_MODELED, "UNKNOWN — REQUIRES VERIFICATION"}


def _clash(e1: Effect, e2: Effect) -> set:
    return set(e1.set_up & e2.set_down) | set(e1.set_down & e2.set_up)


def model_issues(system: SystemModel, actions: ActionUniverse) -> List[str]:
    issues: List[str] = []
    ids = [s.id for s in system.services]
    known = set(ids)

    # ---- services -------------------------------------------------------
    if not ids:
        issues.append("model defines no services")
    if len(ids) != len(known):
        issues.append("duplicate service ids")
    for s in system.services:
        if not s.id or not s.id.strip():
            issues.append("empty service id")

    # ---- dependency rules -----------------------------------------------
    for downstream, rule in system.rules.items():
        where = f"rule[{downstream}]"
        if downstream not in known:
            issues.append(f"{where}: INVALID_DEPENDENCY downstream service does not exist")
        seen = set()
        for inp in rule.inputs:
            if inp.service not in known:
                issues.append(f"{where}: INVALID_DEPENDENCY upstream '{inp.service}' does not exist")
            if inp.service == downstream:
                issues.append(f"{where}: INVALID_DEPENDENCY self-dependency is not allowed")
            if inp.service in seen:
                issues.append(f"{where}: duplicate input '{inp.service}'")
            seen.add(inp.service)
            if inp.topology not in ALLOWED_TOPOLOGY_LABELS:
                issues.append(f"{where}: input '{inp.service}' has unknown topology label '{inp.topology}'")
            if inp.topology == SOURCE_BACKED and not inp.evidence.strip():
                issues.append(f"{where}: input '{inp.service}' is labeled SOURCE-BACKED without evidence")
        if rule.semantics != RIPPLECUT_MODELED:
            issues.append(f"{where}: rule semantics must be labeled RIPPLECUT-MODELED; the MVP never claims "
                          f"a Boolean rule is Online Boutique production semantics")
        hard = rule.hard_upstream
        if rule.rule_type is not RuleType.THRESHOLD and rule.threshold is not None:
            issues.append(f"{where}: 'threshold' is only valid for THRESHOLD rules")
        if rule.rule_type is not RuleType.GROUP and rule.groups:
            issues.append(f"{where}: 'groups' is only valid for GROUP rules")
        if rule.rule_type is RuleType.OR and not hard:
            issues.append(f"{where}: OR rule needs at least one hard input (an empty OR is constantly 0)")
        if rule.rule_type is RuleType.THRESHOLD:
            q = rule.threshold
            if not isinstance(q, int) or isinstance(q, bool) or not (1 <= q <= len(hard)):
                issues.append(f"{where}: THRESHOLD needs an integer 1 <= q <= {len(hard)} (got {q!r})")
        if rule.rule_type is RuleType.GROUP:
            if not rule.groups:
                issues.append(f"{where}: GROUP rule needs at least one group")
            covered = set()
            for g in rule.groups:
                if not g.members:
                    issues.append(f"{where}: empty group")
                if len(set(g.members)) != len(g.members):
                    issues.append(f"{where}: duplicate group member")
                if not set(g.members) <= set(hard):
                    issues.append(f"{where}: group members {sorted(set(g.members) - set(hard))} are not hard inputs")
                if not isinstance(g.q, int) or isinstance(g.q, bool) or not (1 <= g.q <= len(g.members)):
                    issues.append(f"{where}: group q must be an integer in [1, {len(g.members)}] (got {g.q!r})")
                covered |= set(g.members)
            if set(hard) - covered:
                issues.append(f"{where}: hard inputs {sorted(set(hard) - covered)} belong to no group")

    # ---- criticality ------------------------------------------------------
    parts = [system.critical, system.non_critical, system.infrastructure]
    for name, part in zip(("critical", "non_critical", "infrastructure"), parts):
        unknown = part - known
        if unknown:
            issues.append(f"criticality.{name} references unknown services {sorted(unknown)}")
    for a, b in combinations(zip(("critical", "non_critical", "infrastructure"), parts), 2):
        if a[1] & b[1]:
            issues.append(f"services {sorted(a[1] & b[1])} are listed as both {a[0]} and {b[0]}")
    unclassified = known - system.critical - system.non_critical - system.infrastructure
    if unclassified:
        issues.append(f"services {sorted(unclassified)} have no criticality classification")
    role_infra = {s.id for s in system.services if s.role is ServiceRole.INFRASTRUCTURE}
    if role_infra != set(system.infrastructure):
        issues.append("criticality.infrastructure must equal the set of services with role 'infrastructure'")

    # ---- interventions ------------------------------------------------------
    aids = [a.id for a in actions.interventions]
    if len(aids) != len(set(aids)):
        issues.append("duplicate intervention ids")
    for a in actions.interventions:
        where = f"intervention[{a.id}]"
        if not a.id.strip():
            issues.append("empty intervention id")
        if a.cost < 0:
            issues.append(f"{where}: negative cost")
        eff_services = a.effect.services
        if not eff_services:
            issues.append(f"{where}: effect is empty (an action must change at least one service)")
        for sid in set(a.targets) | eff_services:
            if sid not in known:
                issues.append(f"{where}: INVALID_SERVICE target '{sid}' does not exist")
        if set(a.targets) != set(eff_services):
            issues.append(f"{where}: targets {sorted(a.targets)} must equal the services in its effect "
                          f"{sorted(eff_services)}")
        if a.effect.set_up & a.effect.set_down:
            issues.append(f"{where}: effect sets {sorted(a.effect.set_up & a.effect.set_down)} both UP and DOWN")
        for p in a.preconditions:
            if p.service not in known:
                issues.append(f"{where}: precondition references unknown service '{p.service}'")

    # ---- conflicts ----------------------------------------------------------
    known_actions = set(aids)
    for key, c in actions.conflicts.items():
        where = f"conflict{sorted(key)}"
        if not key <= known_actions:
            issues.append(f"{where}: references unknown actions {sorted(key - known_actions)}")
        if c.resolution is ConflictResolution.DEFINED_JOINT_EFFECT:
            j = c.joint_effect
            if j is None or not j.services:
                issues.append(f"{where}: DEFINED_JOINT_EFFECT must define a non-empty joint effect")
            else:
                if j.set_up & j.set_down:
                    issues.append(f"{where}: joint effect sets a service both UP and DOWN")
                for sid in j.services:
                    if sid not in known:
                        issues.append(f"{where}: joint effect references unknown service '{sid}'")

    # ---- undeclared contradictory effects (composition must be explicit) ---
    declared = set(actions.conflicts)
    for a, b in combinations(actions.interventions, 2):
        if frozenset((a.id, b.id)) in declared:
            continue
        clash = _clash(a.effect, b.effect)
        if clash:
            issues.append(f"actions '{a.id}' and '{b.id}' set {sorted(clash)} in opposite directions but no "
                          f"conflict is declared; declare INVALID_COMBINATION or DEFINED_JOINT_EFFECT")

    joint_units = [(k, c.joint_effect) for k, c in actions.conflicts.items()
                   if c.resolution is ConflictResolution.DEFINED_JOINT_EFFECT and c.joint_effect is not None]

    def any_declared(xs, ys) -> bool:
        return any(frozenset((x, y)) in declared for x in xs for y in ys if x != y)

    for pair, joint in joint_units:
        for c in actions.interventions:
            if c.id in pair or any_declared(pair, [c.id]):
                continue
            clash = _clash(joint, c.effect)
            if clash:
                issues.append(f"joint effect of {sorted(pair)} and action '{c.id}' contradict on {sorted(clash)} "
                              f"without a declared conflict")
    for (p1, j1), (p2, j2) in combinations(joint_units, 2):
        if p1 & p2 or any_declared(p1, p2):
            continue
        clash = _clash(j1, j2)
        if clash:
            issues.append(f"joint effects of {sorted(p1)} and {sorted(p2)} contradict on {sorted(clash)}")
    return issues


def validate_model(system: SystemModel, actions: ActionUniverse) -> None:
    issues = model_issues(system, actions)
    if issues:
        raise ModelValidationError(f"model validation failed with {len(issues)} issue(s): " + "; ".join(issues),
                                   details={"issues": issues})
