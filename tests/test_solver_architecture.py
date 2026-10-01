"""Solver interface, registry, policy, guard, orchestrator and fallback (master spec §25-§33, §57-§61,
§117-§121; GATES 8 and 9)."""
import time
from dataclasses import replace
from fractions import Fraction

import pytest
from conftest import action, build, hard, make_problem, rule

from ripplecut.errors import RippleCutError
from ripplecut.model.schema import ResourceLimits
from ripplecut.solvers.base import (OptimalityStatus, Solver, SolverCapabilities, SolverKind, SolverResult,
                                    SolverStatus, result_from_evaluation)
from ripplecut.solvers.branch_and_bound import BranchAndBoundSolver
from ripplecut.solvers.builtin import build_default_registry
from ripplecut.solvers.exhaustive import ExhaustiveSolver
from ripplecut.solvers.fault_injection import FAULT_MODES, FaultInjectingSolver
from ripplecut.solvers.guard import ExecutionGuard
from ripplecut.solvers.orchestrator import SolverOrchestrator
from ripplecut.solvers.policy import SolverPolicy
from ripplecut.solvers.registry import SolverRegistry

FAST = ResourceLimits(timeout_seconds=0.4, max_evaluations=100_000, grace_seconds=0.3)


def policy(order, fallback=(), cross=(), **kw):
    order = tuple(order)
    return SolverPolicy(enabled=order + tuple(fallback), primary=order, fallback=tuple(fallback),
                        limits=kw.pop("limits", FAST), cross_check_solvers=tuple(cross), **kw)


# ---- test solvers ---------------------------------------------------------------------------------------------
class BadSolver(Solver):
    """Master spec §117: proposes an action that does not exist."""
    name = "bad"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        return SolverResult(self.name, SolverStatus.SUCCESS, selected_plan=("launch_rockets",), objective_value=(0, 1, 0),
                            cost=0, intervention_count=1, residual_failures=0,
                            critical_services_preserved=problem.system.c_min, search_complete=True,
                            optimality_status=OptimalityStatus.PROVEN_OPTIMAL)


class CrashSolver(Solver):
    """Master spec §118."""
    name = "crash"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        raise RuntimeError("boom")


class CooperativeSlowSolver(Solver):
    """Master spec §119: exceeds the timeout but honours the context deadline."""
    name = "slow"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        while True:
            context.check_limits()
            time.sleep(0.005)


class StubbornSlowSolver(Solver):
    """Ignores cancellation entirely; the guard must abandon it."""
    name = "stubborn"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        time.sleep(1.5)
        return SolverResult(self.name, SolverStatus.SUCCESS, selected_plan=())


class FixedPlanSolver(Solver):
    """A 'heuristic' that returns a given plan with honestly computed metrics."""
    capabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)

    def __init__(self, name, plan, claim=OptimalityStatus.HEURISTIC):
        self.name, self.plan, self.claim = name, tuple(plan), claim

    def solve(self, problem, context):
        ev = context.evaluate(self.plan)
        return result_from_evaluation(self.name, SolverStatus.SUCCESS, ev, search_complete=False,
                                      optimality=self.claim, nodes_explored=1, metadata={})


class TinySolver(Solver):
    capabilities = SolverCapabilities(kind=SolverKind.EXACT, max_actions=2)
    name = "tiny"

    def solve(self, problem, context):  # pragma: no cover - never started
        raise AssertionError("an incompatible solver must not run")


class MalformedSolver(Solver):
    name = "malformed"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        return {"plan": ["payment_fallback"]}


class LiarInfeasibleSolver(Solver):
    """Claims infeasibility (with a complete-search flag) on a feasible instance."""
    name = "liar_infeasible"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT)

    def solve(self, problem, context):
        return SolverResult(self.name, SolverStatus.NO_FEASIBLE_PLAN, search_complete=True,
                            optimality_status=OptimalityStatus.PROVEN_OPTIMAL)


def payment(bundle):
    return make_problem(bundle, ["paymentservice"], limits=FAST)


def reg(*solvers):
    return SolverRegistry(list(solvers))


# ---- registry and policy ----------------------------------------------------------------------------------------
def test_registry_operations():
    r = build_default_registry()
    assert {"branch_and_bound", "exhaustive"} <= set(r.list_available())
    with pytest.raises(RippleCutError):
        r.register(ExhaustiveSolver())                     # duplicate name
    r2 = r.with_override(CrashSolver())
    assert "crash" in r2 and "crash" not in r            # copies, never mutates the original
    r2.unregister("crash")
    with pytest.raises(RippleCutError):
        r2.get("crash")

    class NoName(Solver):
        capabilities = SolverCapabilities(kind=SolverKind.EXACT)

        def solve(self, problem, context):
            return None
    with pytest.raises(RippleCutError):
        SolverRegistry([NoName()])


def test_policy_execution_order_is_configuration():
    p = SolverPolicy(enabled=("a", "b", "c"), primary=("b",), fallback=("a", "b", "x"))
    assert p.execution_order() == ("b", "a")
    with pytest.raises(RippleCutError):
        SolverOrchestrator(reg(ExhaustiveSolver()), policy(["branch_and_bound"]))


# ---- normal operation -----------------------------------------------------------------------------------------
def test_primary_exact_solver_accepted_and_proven(bundle):
    out = SolverOrchestrator(reg(BranchAndBoundSolver(), ExhaustiveSolver()),
                             policy(["branch_and_bound"], ["exhaustive"], cross=["exhaustive"])).solve(payment(bundle))
    assert out.status == "SUCCESS" and out.plan == ("payment_fallback",) and out.selected_solver == "branch_and_bound"
    assert out.fallback_count == 0 and out.optimality_status is OptimalityStatus.PROVEN_OPTIMAL
    assert out.validation.valid and out.verification["status"] == "EQUIVALENT"


# ---- §117 BadSolver -------------------------------------------------------------------------------------------
def test_bad_solver_rejected_then_fallback(bundle):
    out = SolverOrchestrator(reg(BadSolver(), ExhaustiveSolver()), policy(["bad"], ["exhaustive"])).solve(payment(bundle))
    first = out.attempts[0]
    assert first.outcome == "REJECTED_BY_VALIDATOR" and first.result.status is SolverStatus.INVALID_RESULT
    assert "launch_rockets" in first.validation.reason
    assert out.status == "SUCCESS" and out.selected_solver == "exhaustive" and out.fallback_count == 1
    assert out.plan == ("payment_fallback",)


# ---- §118 CrashSolver -----------------------------------------------------------------------------------------
def test_crash_solver_falls_back(bundle):
    out = SolverOrchestrator(reg(CrashSolver(), ExhaustiveSolver()), policy(["crash"], ["exhaustive"])).solve(payment(bundle))
    assert out.attempts[0].result.status is SolverStatus.ERROR and "boom" in out.attempts[0].result.error
    assert out.status == "SUCCESS" and out.selected_solver == "exhaustive"
    assert out.optimality_status is OptimalityStatus.PROVEN_OPTIMAL


# ---- §119 timeout ------------------------------------------------------------------------------------------------
def test_cooperative_timeout_recorded_and_fallback(bundle):
    out = SolverOrchestrator(reg(CooperativeSlowSolver(), ExhaustiveSolver()),
                             policy(["slow"], ["exhaustive"])).solve(payment(bundle))
    assert out.attempts[0].result.status is SolverStatus.TIMEOUT
    assert out.status == "SUCCESS" and out.selected_solver == "exhaustive"


def test_stubborn_solver_is_abandoned(bundle):
    t0 = time.monotonic()
    res = ExecutionGuard().run(StubbornSlowSolver(), payment(bundle), FAST)
    assert res.status is SolverStatus.TIMEOUT and res.metadata.get("abandoned_thread")
    assert time.monotonic() - t0 < 1.4


# ---- §120 heuristic, valid but non-optimal ------------------------------------------------------------------------
def test_heuristic_valid_but_not_proven_optimal(bundle):
    h = FixedPlanSolver("greedy", ["checkout_backend_standby"])        # valid (5,1,0); optimum is (3,1,0)
    out = SolverOrchestrator(reg(h, ExhaustiveSolver()), policy(["greedy"], ["exhaustive"])).solve(payment(bundle))
    assert out.status == "SUCCESS" and out.validation.valid and out.plan == ("checkout_backend_standby",)
    assert out.optimality_status is OptimalityStatus.HEURISTIC


def test_cross_check_flags_non_optimal_heuristic(bundle):
    h = FixedPlanSolver("greedy", ["checkout_backend_standby"])
    out = SolverOrchestrator(reg(h, ExhaustiveSolver()), policy(["greedy"], cross=["exhaustive"])).solve(payment(bundle))
    assert out.optimality_status is OptimalityStatus.HEURISTIC
    assert out.verification["status"] == "MISMATCH" and out.verification["checks"][0]["better_than_accepted"]


def test_cross_check_can_prove_a_heuristic_optimal(bundle):
    h = FixedPlanSolver("greedy", ["payment_fallback"])
    out = SolverOrchestrator(reg(h, ExhaustiveSolver()), policy(["greedy"], cross=["exhaustive"])).solve(payment(bundle))
    assert out.optimality_status is OptimalityStatus.PROVEN_OPTIMAL and "upgraded" in out.optimality_basis


def test_optimality_claim_is_not_trusted(bundle):
    h = FixedPlanSolver("liar", ["checkout_backend_standby"], claim=OptimalityStatus.PROVEN_OPTIMAL)
    out = SolverOrchestrator(reg(h), policy(["liar"])).solve(payment(bundle))
    assert out.optimality_status is OptimalityStatus.HEURISTIC
    assert any("claim not accepted" in d for d in out.diagnostics)


# ---- other failure modes -------------------------------------------------------------------------------------------
def test_incompatible_solver_is_skipped(bundle):
    out = SolverOrchestrator(reg(TinySolver(), ExhaustiveSolver()), policy(["tiny"], ["exhaustive"])).solve(payment(bundle))
    assert out.attempts[0].result.status is SolverStatus.INCOMPATIBLE and out.selected_solver == "exhaustive"


def test_malformed_result(bundle):
    out = SolverOrchestrator(reg(MalformedSolver(), ExhaustiveSolver()),
                             policy(["malformed"], ["exhaustive"])).solve(payment(bundle))
    assert out.attempts[0].result.status is SolverStatus.INVALID_RESULT and out.selected_solver == "exhaustive"


def test_all_solvers_fail_no_fabricated_plan(bundle):
    out = SolverOrchestrator(reg(CrashSolver(), BadSolver(), MalformedSolver()),
                             policy(["crash"], ["bad", "malformed"])).solve(payment(bundle))
    assert out.status == "NO_VALID_SOLVER_RESULT" and out.plan is None and out.validation is None
    assert out.optimality_status is OptimalityStatus.UNKNOWN and len(out.attempts) == 3


def test_validated_incumbent_after_timeout_policy(bundle):
    class IncumbentThenTimeout(Solver):
        name = "incumbent"
        capabilities = SolverCapabilities(kind=SolverKind.EXACT)

        def solve(self, problem, context):
            ev = context.evaluate(["checkout_backend_standby"])
            r = result_from_evaluation(self.name, SolverStatus.TIMEOUT, ev, search_complete=False,
                                       optimality=OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL, nodes_explored=1,
                                       metadata={})
            return r
    p = payment(bundle)
    off = SolverOrchestrator(reg(IncumbentThenTimeout(), CrashSolver()), policy(["incumbent"], ["crash"])).solve(p)
    assert off.status == "NO_VALID_SOLVER_RESULT" and any("incumbent" in d for d in off.diagnostics)
    on = SolverOrchestrator(reg(IncumbentThenTimeout(), CrashSolver()),
                            policy(["incumbent"], ["crash"], accept_validated_incumbent_if_all_fail=True)).solve(p)
    assert on.status == "SUCCESS" and on.optimality_status is OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL


def test_lying_infeasibility_rejected_or_refuted(bundle):
    p = payment(bundle)
    heuristic_liar = LiarInfeasibleSolver()
    heuristic_liar.capabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)
    out = SolverOrchestrator(reg(heuristic_liar, ExhaustiveSolver()),
                             policy(["liar_infeasible"], ["exhaustive"])).solve(p)
    assert out.attempts[0].outcome == "REJECTED_BY_VALIDATOR" and out.plan == ("payment_fallback",)
    # an "exact" liar passes the uncertified tier, but the exact cross-check refutes it constructively
    out = SolverOrchestrator(reg(LiarInfeasibleSolver(), ExhaustiveSolver()),
                             policy(["liar_infeasible"], cross=["exhaustive"])).solve(p)
    assert out.status == "SUCCESS" and out.plan == ("payment_fallback",) and out.selected_solver == "exhaustive"
    assert out.attempts[0].outcome == "REFUTED_BY_CROSS_CHECK"
    assert out.verification["checks"][0]["verdict"] == "REFUTES_INFEASIBILITY"


@pytest.mark.parametrize("mode", FAULT_MODES)
def test_fault_injection_modes_fall_back(bundle, mode):
    base = build_default_registry()
    r = base.with_override(FaultInjectingSolver(base.get("branch_and_bound"), mode))
    out = SolverOrchestrator(r, policy(["branch_and_bound"], ["exhaustive"])).solve(payment(bundle))
    assert out.attempts[0].outcome in ("FAILED", "REJECTED_BY_VALIDATOR")
    assert out.status == "SUCCESS" and out.selected_solver == "exhaustive" and out.plan == ("payment_fallback",)


# ---- §121 extensibility --------------------------------------------------------------------------------------------
class EmptyPlanSolver(Solver):
    name = "trivial_empty"
    capabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)

    def solve(self, problem, context):
        ev = context.evaluate(())
        if not ev.feasible:
            return SolverResult(self.name, SolverStatus.ERROR, error="empty plan infeasible")
        return result_from_evaluation(self.name, SolverStatus.SUCCESS, ev, search_complete=False,
                                      optimality=OptimalityStatus.HEURISTIC, nodes_explored=1, metadata={})


class BestSingleSolver(Solver):
    name = "trivial_best_single"
    capabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)

    def solve(self, problem, context):
        evs = [context.evaluate((a,)) for a in problem.actions.ids]
        feas = sorted((e for e in evs if e.feasible), key=lambda e: e.key)
        if not feas:
            return SolverResult(self.name, SolverStatus.ERROR, error="no feasible single action")
        return result_from_evaluation(self.name, SolverStatus.SUCCESS, feas[0], search_complete=False,
                                      optimality=OptimalityStatus.HEURISTIC, nodes_explored=len(evs), metadata={})


class ExactPairsSolver(Solver):
    """Third solver: exact only for plans of size <= 2, so it must declare itself HEURISTIC."""
    name = "trivial_pairs"
    capabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)

    def solve(self, problem, context):
        from itertools import combinations
        ids = sorted(problem.actions.ids)
        evs = [context.evaluate(c) for r in (0, 1, 2) for c in combinations(ids, r)]
        feas = sorted((e for e in evs if e.feasible), key=lambda e: e.key)
        if not feas:
            return SolverResult(self.name, SolverStatus.ERROR, error="nothing feasible up to size 2")
        return result_from_evaluation(self.name, SolverStatus.SUCCESS, feas[0], search_complete=False,
                                      optimality=OptimalityStatus.HEURISTIC, nodes_explored=len(evs), metadata={})


def test_extensibility_two_then_three_solvers(bundle):
    p = payment(bundle)
    two = SolverOrchestrator(reg(EmptyPlanSolver(), BestSingleSolver()),
                             policy(["trivial_empty"], ["trivial_best_single"])).solve(p)
    assert two.status == "SUCCESS" and two.selected_solver == "trivial_best_single" and two.fallback_count == 1
    three = SolverOrchestrator(reg(EmptyPlanSolver(), BestSingleSolver(), ExactPairsSolver()),
                               policy(["trivial_empty", "trivial_pairs"], ["trivial_best_single"])).solve(p)
    assert three.selected_solver == "trivial_pairs" and three.plan == ("payment_fallback",)
    assert three.optimality_status is OptimalityStatus.HEURISTIC


def test_thirty_solvers_same_code_path(bundle):
    solvers = [type(f"Crash{i}", (CrashSolver,), {"name": f"crash_{i:02d}"})() for i in range(29)]
    solvers.append(ExhaustiveSolver())
    names = [s.name for s in solvers]
    out = SolverOrchestrator(SolverRegistry(solvers), policy(names[:1], names[1:])).solve(payment(bundle))
    assert out.status == "SUCCESS" and out.selected_solver == "exhaustive" and out.fallback_count == 29
