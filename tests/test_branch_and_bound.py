"""GATE 6 - Branch & Bound equals the exhaustive oracle (master spec §36-§44, §116)."""
import pytest
from conftest import action, build, hard, make_problem, rule
from test_exhaustive import obvious_instance

from ripplecut.engine.objective import evaluate_plan
from ripplecut.generators import random_problem
from ripplecut.solvers.base import SolverKind, SolverStatus
from ripplecut.solvers.branch_and_bound import BranchAndBoundSolver
from ripplecut.solvers.exhaustive import ExhaustiveSolver
from ripplecut.solvers.guard import ExecutionGuard
from ripplecut.validation.validator import SafetyValidator

VARIANTS = [BranchAndBoundSolver("best_first"), BranchAndBoundSolver("depth_first", name="dfs"),
            BranchAndBoundSolver("best_first", name="bf_no_bounds", use_feasibility_bound=False,
                                 use_residual_bound=False),
            BranchAndBoundSolver("depth_first", name="dfs_no_residual", use_residual_bound=False)]


def _outcome(problem, solver):
    res = ExecutionGuard().run(solver, problem)
    if res.status is SolverStatus.NO_FEASIBLE_PLAN:
        return ("INFEASIBLE",), res
    assert res.status is SolverStatus.SUCCESS, (solver.name, res.status, res.error)
    val = SafetyValidator().validate(problem, res, SolverKind.EXACT)
    assert val.valid, val.reason
    o = val.recomputed_objective
    return ((o.cost, o.count, o.residual), res.selected_plan), res


def test_bb_known_optimum_and_validator_accepts():
    """Master spec §116: Exhaustive = expected, B&B = expected, validator accepts both."""
    b, p = obvious_instance()
    for solver in [ExhaustiveSolver()] + VARIANTS:
        out, res = _outcome(p, solver)
        assert out == ((2, 1, 0), ("fix_pay",)), solver.name


@pytest.mark.parametrize("seed", range(400))
def test_bb_equals_exhaustive_random(seed):
    problem, _ = random_problem(seed, n_services=5 + seed % 7, m_actions=2 + seed % 10)
    oracle, _ = _outcome(problem, ExhaustiveSolver())
    for solver in VARIANTS:
        out, res = _outcome(problem, solver)
        assert out == oracle, (seed, solver.name)                 # identical tuple AND identical plan
        assert res.nodes_explored <= 2 ** problem.m


@pytest.mark.parametrize("scenario_failed", [["paymentservice"], ["shippingservice"], ["cartservice"],
                                             ["paymentservice", "shippingservice"], ["emailservice"],
                                             ["productcatalogservice"], ["currencyservice"], []])
def test_bb_equals_exhaustive_canonical(bundle, scenario_failed):
    p = make_problem(bundle, scenario_failed)
    oracle, _ = _outcome(p, ExhaustiveSolver())
    for solver in VARIANTS:
        assert _outcome(p, solver)[0] == oracle


def test_bb_explores_fewer_nodes_on_canonical(bundle):
    _, res = _outcome(make_problem(bundle, ["paymentservice"]), BranchAndBoundSolver())
    assert res.nodes_explored < 256 and res.metadata["precondition_excluded_actions"]


def test_naive_count_bound_is_unsafe():
    """LB_N = N + ceil(critical failures / max single-action recoveries) is NOT a valid bound under AND rules.

    X = A AND B is critical; A and B are DOWN. Each single action restores zero critical services, so the naive
    per-action recovery rate is 0 and the naive bound would declare every plan infeasible (or divide by zero).
    Yet {fa, fb} restores X: recoveries are super-additive. RippleCut therefore does not use this bound.
    """
    b = build(services=["A", "B", "X"], critical=["X"], rules=[rule("X", "AND", hard("A", "B"))],
              actions=[action("fa", 1, up=["A"]), action("fb", 1, up=["B"])])
    p = make_problem(b, state={"A": 0, "B": 0, "X": 1})
    base = evaluate_plan(p, []).objective.critical_up
    single = [evaluate_plan(p, [a]).objective.critical_up - base for a in ("fa", "fb")]
    joint = evaluate_plan(p, ["fa", "fb"]).objective.critical_up - base
    assert max(single) == 0 and joint == 1                            # super-additive
    for solver in [ExhaustiveSolver()] + VARIANTS:
        out, _ = _outcome(p, solver)
        assert out == ((2, 2, 0), ("fa", "fb"))


def test_bb_timeout_returns_validated_incumbent_status():
    problem, _ = random_problem(7, n_services=12, m_actions=18)
    from ripplecut.model.schema import ResourceLimits
    res = ExecutionGuard().run(BranchAndBoundSolver("depth_first"), problem,
                               ResourceLimits(timeout_seconds=5, max_evaluations=3, grace_seconds=0.5))
    assert res.status in (SolverStatus.RESOURCE_LIMIT, SolverStatus.NO_FEASIBLE_PLAN, SolverStatus.SUCCESS)
    if res.status is SolverStatus.RESOURCE_LIMIT:
        assert not res.search_complete


def test_multiple_equally_optimal_plans_resolved_by_sorted_ids():
    """Master spec §64 case 9 / §45: identical (K, N, R) -> the lexicographically smallest sorted id tuple."""
    b = build(services=["C"], critical=["C"], actions=[action("zeta", 1, up=["C"]), action("alpha", 1, up=["C"]),
                                                          action("mid", 1, up=["C"])])
    p = make_problem(b, ["C"])
    for solver in [ExhaustiveSolver()] + VARIANTS:
        out, _ = _outcome(p, solver)
        assert out == ((1, 1, 0), ("alpha",)), solver.name
