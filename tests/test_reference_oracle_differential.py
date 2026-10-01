"""Differential testing: Reference Oracle vs Exhaustive vs Branch & Bound (Phase 1, master spec §1.4).

Verifies that for every supported rule type (AND, OR, THRESHOLD, GROUP), soft inputs,
conflicts, preconditions, joint effects, cycles, already-safe cases, and infeasible cases:
1. Reference Oracle
2. Exhaustive Solver
3. Branch & Bound Solver (both best_first and depth_first)
agree on:
- Feasibility
- Selected optimal plan
- Objective tuple (K, N, R)
- Critical services preserved
- Tie-breaking behavior
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple

import pytest

from conftest import action, build, hard, make_problem, rule, soft
from reference_oracle import reference_solve
from ripplecut.generators import random_problem
from ripplecut.model.schema import (
    ContainmentProblem,
)
from ripplecut.solvers.branch_and_bound import BranchAndBoundSolver
from ripplecut.solvers.exhaustive import ExhaustiveSolver
from ripplecut.solvers.guard import ExecutionGuard


def _solve_with_solvers(problem: ContainmentProblem):
    guard = ExecutionGuard()
    res_ex = guard.run(ExhaustiveSolver(), problem)
    res_bnb_bf = guard.run(BranchAndBoundSolver(strategy="best_first"), problem)
    res_bnb_df = guard.run(BranchAndBoundSolver(strategy="depth_first"), problem)

    oracle = reference_solve(problem)

    return oracle, res_ex, res_bnb_bf, res_bnb_df


def _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df):
    # Status agreement
    assert res_ex.status.value == oracle.status, f"Exhaustive status mismatch: {res_ex.status} vs {oracle.status}"
    assert res_bnb_bf.status.value == oracle.status, f"B&B BF status mismatch: {res_bnb_bf.status} vs {oracle.status}"
    assert res_bnb_df.status.value == oracle.status, f"B&B DF status mismatch: {res_bnb_df.status} vs {oracle.status}"

    if oracle.status == "SUCCESS":
        assert tuple(res_ex.selected_plan) == oracle.best_plan
        assert tuple(res_bnb_bf.selected_plan) == oracle.best_plan
        assert tuple(res_bnb_df.selected_plan) == oracle.best_plan

        # Objective (K, N, R)
        oracle_k, oracle_n, oracle_r = oracle.best_objective
        assert res_ex.cost == oracle_k
        assert res_ex.intervention_count == oracle_n
        assert res_ex.residual_failures == oracle_r

        assert res_bnb_bf.cost == oracle_k
        assert res_bnb_bf.intervention_count == oracle_n
        assert res_bnb_bf.residual_failures == oracle_r

        assert res_bnb_df.cost == oracle_k
        assert res_bnb_df.intervention_count == oracle_n
        assert res_bnb_df.residual_failures == oracle_r

        assert res_ex.critical_services_preserved == oracle.critical_preserved
        assert res_bnb_bf.critical_services_preserved == oracle.critical_preserved
        assert res_bnb_df.critical_services_preserved == oracle.critical_preserved


def test_oracle_vs_solvers_and_rules():
    """AND rule cascade with conflicting and non-conflicting interventions."""
    b = build(
        services=["s1", "s2", "s3"],
        rules=[rule("s3", "AND", hard("s1", "s2"))],
        critical=["s1", "s2", "s3"],
        actions=[
            action("fix_s1", 2, up=["s1"], pre=[("s1", "DOWN")]),
            action("fix_s2", 3, up=["s2"], pre=[("s2", "DOWN")]),
            action("fix_both", 4, up=["s1", "s2"]),
        ],
    )
    # s1 and s2 initially DOWN => s3 also fails in cascade
    problem = make_problem(b, failed=["s1", "s2"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    assert oracle.best_plan == ("fix_both",)  # Cost 4 beats 2+3=5


def test_oracle_vs_solvers_or_rules():
    """OR rule: s3 operational if s1 OR s2 is UP."""
    b = build(
        services=["s1", "s2", "s3"],
        rules=[rule("s3", "OR", hard("s1", "s2"))],
        critical=["s3"],
        actions=[
            action("fix_s1", 5, up=["s1"]),
            action("fix_s2", 2, up=["s2"]),
        ],
    )
    problem = make_problem(b, failed=["s1", "s2"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    assert oracle.best_plan == ("fix_s2",)  # Cost 2


def test_oracle_vs_solvers_threshold_rules():
    """THRESHOLD rule: s4 needs at least 2 of {s1, s2, s3}."""
    b = build(
        services=["s1", "s2", "s3", "s4"],
        rules=[rule("s4", "THRESHOLD", hard("s1", "s2", "s3"), threshold=2)],
        critical=["s4"],
        actions=[
            action("fix_s1", 3, up=["s1"]),
            action("fix_s2", 2, up=["s2"]),
            action("fix_s3", 4, up=["s3"]),
        ],
    )
    problem = make_problem(b, failed=["s1", "s2", "s3"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    assert oracle.best_plan == ("fix_s1", "fix_s2")  # Cost 3+2=5 beats 2+4=6 and 3+4=7


def test_oracle_vs_solvers_group_rules():
    """GROUP rule: conjunction of threshold groups."""
    b = build(
        services=["s1", "s2", "s3", "s4", "s5"],
        rules=[{
            "downstream": "s5",
            "rule_type": "GROUP",
            "semantics": "RIPPLECUT-MODELED",
            "inputs": hard("s1", "s2", "s3", "s4"),
            "groups": [
                {"members": ["s1", "s2"], "q": 1},
                {"members": ["s3", "s4"], "q": 1},
            ],
        }],
        critical=["s5"],
        actions=[
            action("fix_s1", 3, up=["s1"]),
            action("fix_s2", 4, up=["s2"]),
            action("fix_s3", 2, up=["s3"]),
            action("fix_s4", 5, up=["s4"]),
        ],
    )
    problem = make_problem(b, failed=["s1", "s2", "s3", "s4"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    assert oracle.best_plan == ("fix_s1", "fix_s3")  # Cost 3+2=5


def test_oracle_vs_solvers_cycles_and_soft_deps():
    """Cycle s1 <-> s2 and soft input s3."""
    b = build(
        services=["s1", "s2", "s3"],
        rules=[
            rule("s1", "AND", hard("s2") + soft("s3")),
            rule("s2", "AND", hard("s1")),
        ],
        critical=["s1", "s2"],
        actions=[
            action("fix_s1", 2, up=["s1"]),
            action("fix_s2", 2, up=["s2"]),
            action("fix_s3", 1, up=["s3"]),
        ],
    )
    # s1 and s2 both DOWN. Fixing only s1 still collapses because s2 is DOWN. Both needed.
    # s3 is soft, so fixing s3 alone does not rescue s1 or s2.
    problem = make_problem(b, failed=["s1", "s2", "s3"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    assert oracle.best_plan == ("fix_s1", "fix_s2")


def test_oracle_vs_solvers_conflicts_and_joint_effects():
    """INVALID_COMBINATION and DEFINED_JOINT_EFFECT."""
    b = build(
        services=["s1", "s2"],
        rules=[],
        critical=["s1", "s2"],
        actions=[
            action("a1", 1, up=["s1"]),
            action("a2", 2, up=["s2"]),
            action("a3", 1, up=["s1", "s2"]),
        ],
        conflicts=[
            {"actions": ["a1", "a3"], "resolution": "INVALID_COMBINATION", "rationale": "mutually exclusive"},
            {"actions": ["a1", "a2"], "resolution": "DEFINED_JOINT_EFFECT", "rationale": "joint effect",
             "joint_effect": {"set_up": ["s1"], "set_down": ["s2"]}},
        ],
    )
    problem = make_problem(b, failed=["s1", "s2"])

    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
    # {a1, a2} sets s2 DOWN by joint effect, failing critical requirement for s2.
    # {a1, a3} is INVALID_COMBINATION.
    # {a3} alone succeeds with cost 1.
    assert oracle.best_plan == ("a3",)


def test_oracle_vs_solvers_infeasible_and_already_safe():
    """Infeasible case vs already safe case."""
    b = build(services=["s1"], rules=[], critical=["s1"], actions=[])

    # Infeasible: no actions, service is DOWN
    prob_infeasible = make_problem(b, failed=["s1"])
    oracle_inf, ex_inf, bnb_bf_inf, bnb_df_inf = _solve_with_solvers(prob_infeasible)
    _assert_agreement(oracle_inf, ex_inf, bnb_bf_inf, bnb_df_inf)
    assert oracle_inf.status == "NO_FEASIBLE_PLAN"

    # Already safe: service is UP
    prob_safe = make_problem(b, failed=[])
    oracle_safe, ex_safe, bnb_bf_safe, bnb_df_safe = _solve_with_solvers(prob_safe)
    _assert_agreement(oracle_safe, ex_safe, bnb_bf_safe, bnb_df_safe)
    assert oracle_safe.status == "SUCCESS"
    assert oracle_safe.best_plan == ()


@pytest.mark.parametrize("seed", list(range(10)))
def test_oracle_vs_solvers_randomized(seed: int):
    """Randomized small instances: Oracle vs Exhaustive vs B&B."""
    problem, _ = random_problem(seed=seed, n_services=5, m_actions=5, n_failures=(1, 2), plant_restorers=True)
    oracle, res_ex, res_bnb_bf, res_bnb_df = _solve_with_solvers(problem)
    _assert_agreement(oracle, res_ex, res_bnb_bf, res_bnb_df)
