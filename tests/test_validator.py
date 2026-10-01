"""GATE 7 - the independent validator catches intentionally corrupted results (master spec §34-§35, §56)."""
from dataclasses import replace
from fractions import Fraction

import pytest
from conftest import action, build, make_problem

from ripplecut.solvers.base import OptimalityStatus, SolverKind, SolverResult, SolverStatus
from ripplecut.solvers.exhaustive import ExhaustiveSolver
from ripplecut.solvers.guard import ExecutionGuard
from ripplecut.validation.validator import SafetyValidator

V = SafetyValidator()


@pytest.fixture
def good(bundle):
    p = make_problem(bundle, ["paymentservice"])
    return p, ExecutionGuard().run(ExhaustiveSolver(), p)


def _failed_checks(v):
    return {c.name for c in v.checks if not c.passed}


def test_valid_plan_accepted(good):
    p, res = good
    v = V.validate(p, res, SolverKind.EXACT)
    assert v.valid and len(v.checks) == 10 and all(c.passed for c in v.checks)
    assert v.recomputed_objective.cost == 3 and v.recomputed_state == p.system.all_up()


@pytest.mark.parametrize("corruption,check", [
    (dict(cost=Fraction(2)), "7_cost"),
    (dict(cost="three"), "7_cost"),
    (dict(intervention_count=2), "8_intervention_count"),
    (dict(residual_failures=5), "9_residual_failures"),
    (dict(critical_services_preserved=6), "10_reported_vs_recomputed"),
    (dict(objective_value=(Fraction(1), 1, 0)), "10_reported_vs_recomputed"),
])
def test_wrong_reported_values_rejected(good, corruption, check):
    p, res = good
    v = V.validate(p, replace(res, **corruption), SolverKind.EXACT)
    assert not v.valid and check in _failed_checks(v)


def test_wrong_final_state_rejected(good):
    p, res = good
    bad_state = dict(res.final_state, frontend=0)
    v = V.validate(p, replace(res, final_state=bad_state), SolverKind.EXACT)
    assert not v.valid and "final state" in v.reason


@pytest.mark.parametrize("plan,reason", [
    (("launch_rockets",), "outside the configured universe"),
    (("restart_checkoutservice",), "PRECONDITION_UNMET"),
    (("checkout_backend_standby", "payment_fallback"), "ACTION_CONFLICT"),
    (("payment_fallback", "payment_fallback"), "INVALID_ACTION"),
    ((), "does not preserve every designated critical service"),
])
def test_illegal_or_infeasible_plans_rejected(good, plan, reason):
    p, res = good
    v = V.validate(p, replace(res, selected_plan=plan), SolverKind.EXACT)
    assert not v.valid and reason in v.reason


def test_missing_plan_rejected(good):
    p, res = good
    assert not V.validate_plan(p, replace(res, selected_plan=None)).valid


def test_certified_infeasibility(bundle):
    p = make_problem(bundle, ["cartservice"])
    claim = SolverResult("anyone", SolverStatus.NO_FEASIBLE_PLAN, search_complete=False)
    v = V.validate(p, claim, SolverKind.HEURISTIC)           # certificate does not depend on who claims it
    assert v.valid and v.infeasibility_certified


def conflict_only_infeasible():
    """Two critical services, each restorable by one action, but the two actions may not be combined."""
    return build(services=["a", "b"], critical=["a", "b"],
                 actions=[action("fa", 1, up=["a"]), action("fb", 1, up=["b"])],
                 conflicts=[{"actions": ["fa", "fb"], "resolution": "INVALID_COMBINATION", "rationale": "test"}])


def test_uncertifiable_infeasibility_needs_complete_exact_search():
    b = conflict_only_infeasible()
    p = make_problem(b, ["a", "b"])
    assert not V.infeasibility_certificate(p)["certified"]          # the monotone bound cannot see conflicts
    exact = ExecutionGuard().run(ExhaustiveSolver(), p)
    assert exact.status is SolverStatus.NO_FEASIBLE_PLAN
    v = V.validate(p, exact, SolverKind.EXACT)
    assert v.valid and v.infeasibility_certified is False            # accepted, explicitly uncertified
    assert not V.validate(p, exact, SolverKind.HEURISTIC).valid
    assert not V.validate(p, replace(exact, search_complete=False), SolverKind.EXACT).valid
    strict = SafetyValidator(accept_uncertified_infeasibility_from_exact_solver=False)
    assert not strict.validate(p, exact, SolverKind.EXACT).valid


def test_validator_consistency_property(bundle):
    """For every accepted plan the reported objective equals the recomputed objective (master spec §63)."""
    for failed in (["paymentservice"], ["shippingservice"], ["productcatalogservice"], ["emailservice"]):
        p = make_problem(bundle, failed)
        res = ExecutionGuard().run(ExhaustiveSolver(), p)
        v = V.validate(p, res, SolverKind.EXACT)
        assert v.valid
        o = v.recomputed_objective
        assert (Fraction(res.cost), res.intervention_count, res.residual_failures) == (o.cost, o.count, o.residual)
