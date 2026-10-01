"""Canonical Reference Oracle for RippleCut Mathematical Core (Phase 1, master spec §1.2-§1.3).

This module is an independent, deliberately simple reference implementation used
ONLY for verification and differential testing. It is outside production B&B
and Exhaustive solvers.

It provides a transparent, zero-optimization evaluation of candidate intervention
subsets over small action spaces to certify that both native B&B and the
Exhaustive oracle produce identical results across all supported rule types:
- AND rules
- OR rules
- THRESHOLD rules
- GROUP rules
- Soft (non-critical, non-propagating) dependencies
- Preconditions
- Action conflicts (INVALID_COMBINATION and DEFINED_JOINT_EFFECT)
- Cycles
- Empty plan (already safe)
- Infeasible problems
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ripplecut.model.schema import (
    ConflictResolution,
    ContainmentProblem,
    RuleType,
    State,
    SystemModel,
)


@dataclass(frozen=True)
class OraclePlanResult:
    plan: Tuple[str, ...]
    cost: Fraction
    count: int
    residual: int
    critical_up: int
    feasible: bool
    final_state: State

    @property
    def key(self) -> Tuple[Fraction, int, int, Tuple[str, ...]]:
        return (self.cost, self.count, self.residual, self.plan)


@dataclass(frozen=True)
class OracleOutcome:
    status: str                         # "SUCCESS" or "NO_FEASIBLE_PLAN"
    best_plan: Optional[Tuple[str, ...]]
    best_objective: Optional[Tuple[Fraction, int, int]]  # (K, N, R)
    critical_preserved: int
    total_candidates: int
    legal_candidates: int
    feasible_candidates: int
    all_feasible: Tuple[OraclePlanResult, ...]
    final_state: Optional[State]


def reference_rule_eval(rule: Any, system: SystemModel, state: State) -> int:
    """Independent evaluation of R_v(x) for AND, OR, THRESHOLD, GROUP."""
    idx = system.index
    t = rule.rule_type
    hard_indices = [idx(s) for s in rule.hard_upstream]

    if t is RuleType.AND:
        return 1 if all(state[i] == 1 for i in hard_indices) else 0

    if t is RuleType.OR:
        return 1 if any(state[i] == 1 for i in hard_indices) else 0

    if t is RuleType.THRESHOLD:
        k = int(rule.threshold or 1)
        return 1 if sum(state[i] for i in hard_indices) >= k else 0

    if t is RuleType.GROUP:
        for g in rule.groups:
            g_indices = [idx(m) for m in g.members]
            if sum(state[i] for i in g_indices) < g.q:
                return 0
        return 1

    raise ValueError(f"Unknown rule type: {t}")


def reference_cascade(system: SystemModel, initial_state: State, max_rounds: Optional[int] = None) -> State:
    """Independent synchronous cascade simulation: Phi(x)_v = x_v AND R_v(x)."""
    n = len(system.services)
    limit = max_rounds if max_rounds is not None else n + 2
    x = tuple(initial_state)

    for _ in range(limit):
        nxt = list(x)
        for rule in system.rules.values():
            v_idx = system.index(rule.downstream)
            if x[v_idx] == 1:
                r_val = reference_rule_eval(rule, system, x)
                if r_val == 0:
                    nxt[v_idx] = 0
        nxt_tuple = tuple(nxt)
        if nxt_tuple == x:
            return x
        x = nxt_tuple

    raise RuntimeError(f"Reference cascade did not reach fixed point within {limit} rounds")


def reference_check_plan_legality(
    problem: ContainmentProblem, plan: Sequence[str]
) -> Tuple[bool, Optional[str], Set[str], Set[str]]:
    """Independent plan legality check.

    Returns: (is_legal, reason_if_illegal, set_up_services, set_down_services)
    """
    system = problem.system
    universe = problem.actions
    x0 = problem.initial_state
    chosen = set(plan)

    # 1. Action existence
    for a in chosen:
        if universe.get(a) is None:
            return False, f"Unknown action: {a}", set(), set()

    # 2. Preconditions on x0
    for a in chosen:
        act = universe.get(a)
        for pre in act.preconditions:
            if x0[system.index(pre.service)] != pre.state:
                return False, f"Precondition failed for {a}: {pre.service}", set(), set()

    # 3. Conflicts
    for key, c in universe.conflicts.items():
        if key <= chosen:
            if c.resolution is ConflictResolution.INVALID_COMBINATION:
                return False, f"Invalid conflict combination: {key}", set(), set()

    # 4. Joint effect overlap check
    joint_pairs = [c for key, c in universe.conflicts.items() if key <= chosen and c.resolution is ConflictResolution.DEFINED_JOINT_EFFECT]
    seen_in_joint: Set[str] = set()
    for c in joint_pairs:
        if seen_in_joint & c.actions:
            return False, "Action participates in multiple joint effects", set(), set()
        seen_in_joint |= c.actions

    # 5. Compute UP and DOWN sets
    up: Set[str] = set()
    down: Set[str] = set()
    for c in joint_pairs:
        if c.joint_effect:
            up |= c.joint_effect.set_up
            down |= c.joint_effect.set_down

    for a in chosen - seen_in_joint:
        act = universe.get(a)
        up |= act.effect.set_up
        down |= act.effect.set_down

    if up & down:
        return False, f"Conflicting UP and DOWN effects: {up & down}", set(), set()

    return True, None, up, down


def reference_solve(problem: ContainmentProblem) -> OracleOutcome:
    """Enumerate all 2^m subsets independently and compute optimal lexmin plan.

    Lexmin objective order:
    1. Feasible (C(B) >= C_min)
    2. Minimise total cost K(B)
    3. Minimise action count N(B)
    4. Minimise residual failures R(B)
    5. Lexicographical tie-breaker on sorted action IDs
    """
    system = problem.system
    universe = problem.actions
    c_min = system.c_min
    all_actions = sorted(universe.ids)
    m = len(all_actions)

    feasible_results: List[OraclePlanResult] = []
    total_candidates = 2 ** m
    legal_candidates = 0

    for r in range(m + 1):
        for combo in combinations(all_actions, r):
            plan_tuple = tuple(sorted(combo))
            legal, reason, up, down = reference_check_plan_legality(problem, plan_tuple)
            if not legal:
                continue
            legal_candidates += 1

            # Transform state T_B(x0)
            x_post = list(problem.initial_state)
            for s in up:
                x_post[system.index(s)] = 1
            for s in down:
                x_post[system.index(s)] = 0

            # Run cascade
            x_star = reference_cascade(system, tuple(x_post))

            # Evaluate objective
            down_services = [s for s in system.service_ids if x_star[system.index(s)] == 0]
            crit_down = [s for s in down_services if s in system.critical]
            crit_up = len(system.critical) - len(crit_down)
            is_feasible = crit_up >= c_min

            cost = sum((universe.get(a).cost for a in plan_tuple), Fraction(0))
            count = len(plan_tuple)
            residual = len(down_services)

            res = OraclePlanResult(
                plan=plan_tuple,
                cost=cost,
                count=count,
                residual=residual,
                critical_up=crit_up,
                feasible=is_feasible,
                final_state=x_star,
            )

            if is_feasible:
                feasible_results.append(res)

    feasible_results.sort(key=lambda r: r.key)

    if not feasible_results:
        return OracleOutcome(
            status="NO_FEASIBLE_PLAN",
            best_plan=None,
            best_objective=None,
            critical_preserved=0,
            total_candidates=total_candidates,
            legal_candidates=legal_candidates,
            feasible_candidates=0,
            all_feasible=(),
            final_state=None,
        )

    best = feasible_results[0]
    return OracleOutcome(
        status="SUCCESS",
        best_plan=best.plan,
        best_objective=(best.cost, best.count, best.residual),
        critical_preserved=best.critical_up,
        total_candidates=total_candidates,
        legal_candidates=legal_candidates,
        feasible_candidates=len(feasible_results),
        all_feasible=tuple(feasible_results),
        final_state=best.final_state,
    )
