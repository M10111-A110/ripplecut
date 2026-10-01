"""The single RippleCut dependency engine (master spec §12, PDF §6.1).

For a downstream service v with hard inputs D_v = {u_1..u_k}:

    AND        R_v(x) = prod_i x_{u_i}                  (empty product = 1: a root)
    OR         R_v(x) = 1[ sum_i x_{u_i} >= 1 ]
    THRESHOLD  R_v(x) = 1[ sum_i x_{u_i} >= q ]
    GROUP      R_v(x) = prod_g 1[ sum_{u in g} x_u >= q_g ]    (conjunction of threshold groups)

Soft inputs are recorded topology but are never read by R_v, so a soft
upstream failure never makes v unavailable. Services without a configured
rule have R_v = 1.

Every supported rule is monotone: x <= y (componentwise) implies R_v(x) <= R_v(y).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from ..model.schema import DependencyRule, RuleType, State, SystemModel


@dataclass(frozen=True)
class CompiledRule:
    downstream_index: int
    rule_type: RuleType
    hard: Tuple[int, ...]
    threshold: int
    groups: Tuple[Tuple[Tuple[int, ...], int], ...]

    def evaluate(self, x: State) -> int:
        t = self.rule_type
        if t is RuleType.AND:
            for i in self.hard:
                if x[i] == 0:
                    return 0
            return 1
        if t is RuleType.OR:
            for i in self.hard:
                if x[i] == 1:
                    return 1
            return 0
        if t is RuleType.THRESHOLD:
            return 1 if sum(x[i] for i in self.hard) >= self.threshold else 0
        if t is RuleType.GROUP:
            for members, q in self.groups:
                if sum(x[i] for i in members) < q:
                    return 0
            return 1
        raise ValueError(f"unsupported rule type {t}")  # unreachable after model validation


def compile_rule(system: SystemModel, rule: DependencyRule) -> CompiledRule:
    idx = system.index
    return CompiledRule(
        downstream_index=idx(rule.downstream),
        rule_type=rule.rule_type,
        hard=tuple(idx(s) for s in rule.hard_upstream),
        threshold=int(rule.threshold or 0),
        groups=tuple((tuple(idx(m) for m in g.members), g.q) for g in rule.groups),
    )


def compiled_rules(system: SystemModel) -> Tuple[CompiledRule, ...]:
    """Compile once per (immutable) system model and cache on the object."""
    cached = system.__dict__.get("_compiled_rules")
    if cached is None:
        cached = tuple(compile_rule(system, r) for r in system.rules.values())
        object.__setattr__(system, "_compiled_rules", cached)
    return cached


def evaluate_rule(system: SystemModel, rule: DependencyRule, state: State) -> int:
    return compile_rule(system, rule).evaluate(state)


def rule_values(system: SystemModel, state: State) -> Dict[str, int]:
    """R_v(x) for every service (1 for services without a rule)."""
    out = {sid: 1 for sid in system.service_ids}
    for cr in compiled_rules(system):
        out[system.service_ids[cr.downstream_index]] = cr.evaluate(state)
    return out


def unsatisfied_inputs(system: SystemModel, service_id: str, state: State) -> List[str]:
    """Hard inputs of ``service_id`` that are DOWN in ``state`` (used for explanations only)."""
    rule = system.rules.get(service_id)
    if rule is None:
        return []
    return [u for u in rule.hard_upstream if state[system.index(u)] == 0]


def describe_rule(rule: DependencyRule) -> str:
    hard = list(rule.hard_upstream)
    if rule.rule_type is RuleType.AND:
        body = " AND ".join(hard) if hard else "1 (no hard inputs)"
    elif rule.rule_type is RuleType.OR:
        body = " OR ".join(hard)
    elif rule.rule_type is RuleType.THRESHOLD:
        body = f"at least {rule.threshold} of {{{', '.join(hard)}}}"
    else:
        body = " AND ".join(f"[at least {g.q} of {{{', '.join(g.members)}}}]" for g in rule.groups)
    soft = list(rule.soft_upstream)
    tail = f"   (soft, non-propagating: {', '.join(soft)})" if soft else ""
    return f"{rule.downstream} = {body}{tail}"


def rules_summary(system: SystemModel) -> Mapping[str, str]:
    return {r.downstream: describe_rule(r) for r in system.rules.values()}
