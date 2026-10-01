"""GATE 5 - exhaustive oracle on known examples (master spec §42-§43, §116)."""
from conftest import action, build, hard, make_problem, rule

from ripplecut.solvers.base import OptimalityStatus, SolverKind, SolverStatus
from ripplecut.solvers.exhaustive import ExhaustiveSolver
from ripplecut.solvers.guard import ExecutionGuard
from ripplecut.validation.validator import SafetyValidator


def obvious_instance():
    """checkout = pay AND ship; pay DOWN. Only 'fix_pay' (2) restores it; 'big' (9) also does; 'noise' irrelevant."""
    b = build(services=["pay", "ship", "checkout", "email"], critical=["pay", "ship", "checkout"],
              rules=[rule("checkout", "AND", hard("pay", "ship"))],
              actions=[action("fix_pay", 2, up=["pay"], pre=[("pay", "DOWN")]), action("big", 9, up=["pay", "ship"]),
                       action("noise", 0, up=["email"])])
    return b, make_problem(b, ["pay"])


def test_exhaustive_known_optimum():
    b, p = obvious_instance()
    res = ExecutionGuard().run(ExhaustiveSolver(), p)
    assert res.status is SolverStatus.SUCCESS and res.selected_plan == ("fix_pay",)
    assert (res.cost, res.intervention_count, res.residual_failures) == (2, 1, 0)
    assert res.search_complete and res.metadata["candidates_total"] == 8
    assert SafetyValidator().validate(p, res, SolverKind.EXACT).valid
    tops = res.metadata["top_feasible"]
    assert [t["plan"] for t in tops[:2]] == [["fix_pay"], ["fix_pay", "noise"]]


def test_exhaustive_infeasible():
    b = build(services=["a", "c"], critical=["a", "c"], rules=[rule("c", "AND", hard("a"))],
              actions=[action("x", 1, up=["c"])])
    res = ExecutionGuard().run(ExhaustiveSolver(), make_problem(b, ["a"]))
    assert res.status is SolverStatus.NO_FEASIBLE_PLAN and res.selected_plan is None and res.search_complete
    assert res.optimality_status is OptimalityStatus.PROVEN_OPTIMAL


def test_no_interventions_configured():
    b = build(services=["a"], critical=["a"])
    res = ExecutionGuard().run(ExhaustiveSolver(), make_problem(b, []))
    assert res.status is SolverStatus.SUCCESS and res.selected_plan == ()
    res = ExecutionGuard().run(ExhaustiveSolver(), make_problem(b, ["a"]))
    assert res.status is SolverStatus.NO_FEASIBLE_PLAN


def test_exhaustive_declares_its_limit():
    b = build(services=["a"], critical=["a"], actions=[action(f"x{i:02d}", 1, up=["a"]) for i in range(17)])
    res = ExecutionGuard().run(ExhaustiveSolver(), make_problem(b, ["a"]))
    assert res.status is SolverStatus.INCOMPATIBLE


def test_canonical_payment_oracle(bundle):
    res = ExecutionGuard().run(ExhaustiveSolver(), make_problem(bundle, ["paymentservice"]))
    assert res.selected_plan == ("payment_fallback",) and res.metadata["candidates_total"] == 128
    assert res.metadata["top_feasible"][1]["plan"] == ["checkout_backend_standby"]
