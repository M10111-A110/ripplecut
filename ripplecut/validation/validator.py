"""Independent Safety Validator (master spec §34-§35, §56). "Solver proposes. RippleCut verifies."

The validator never trusts a solver's reported numbers. For a proposed plan it
re-runs RippleCut's own intervention engine, cascade simulator and objective
evaluator from the problem definition, then compares every reported metric.
Any mismatch makes the result INVALID_RESULT.

Infeasibility claims (NO_FEASIBLE_PLAN) cannot be refuted by replaying a single
plan, so they are handled in two tiers:

  1. Certificate. Apply every UP effect of every precondition-satisfiable action
     (and every defined joint effect) to x(0), ignore all DOWN effects, and run
     the cascade. By monotonicity this state dominates T_B(x(0)) for every
     legal B, so if even it violates the critical constraint, infeasibility is
     PROVEN independently of any solver.
  2. Otherwise the claim is accepted only from an EXACT solver whose search
     completed, and only if policy allows it; it is labeled "uncertified".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional

from ..engine.interventions import check_plan, optimistic_state, statically_applicable, transform
from ..engine.objective import ObjectiveValue, compute_objective
from ..engine.simulator import simulate
from ..model.schema import ContainmentProblem, State, cost_to_json
from ..solvers.base import SolverKind, SolverResult, SolverStatus


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    checks: List[ValidationCheck] = field(default_factory=list)
    recomputed_objective: Optional[ObjectiveValue] = None
    recomputed_state: Optional[State] = None
    infeasibility_certified: Optional[bool] = None

    def to_dict(self, problem: Optional[ContainmentProblem] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"valid": self.valid, "reason": self.reason,
                               "checks": [c.to_dict() for c in self.checks],
                               "recomputed_objective": self.recomputed_objective.to_dict()
                               if self.recomputed_objective else None,
                               "infeasibility_certified": self.infeasibility_certified}
        if problem is not None and self.recomputed_state is not None:
            out["recomputed_final_state"] = problem.system.state_to_mapping(self.recomputed_state)
        return out


def _as_fraction(value: Any) -> Optional[Fraction]:
    if isinstance(value, bool):
        return None
    try:
        return Fraction(str(value)) if not isinstance(value, Fraction) else value
    except (ValueError, ZeroDivisionError, TypeError):
        return None


class SafetyValidator:
    def __init__(self, accept_uncertified_infeasibility_from_exact_solver: bool = True) -> None:
        self.accept_uncertified_infeasibility = accept_uncertified_infeasibility_from_exact_solver

    # ------------------------------------------------------------------------
    def validate(self, problem: ContainmentProblem, result: SolverResult,
                 solver_kind: SolverKind = SolverKind.HEURISTIC) -> ValidationResult:
        if result.status is SolverStatus.NO_FEASIBLE_PLAN:
            return self.validate_infeasibility(problem, result, solver_kind)
        return self.validate_plan(problem, result)

    # ------------------------------------------------------------------------
    def validate_plan(self, problem: ContainmentProblem, result: SolverResult) -> ValidationResult:
        system, universe = problem.system, problem.actions
        checks: List[ValidationCheck] = []

        def fail(reason: str) -> ValidationResult:
            return ValidationResult(False, reason, checks)

        plan = result.selected_plan
        ok = plan is not None and isinstance(plan, (tuple, list)) and all(isinstance(a, str) for a in plan)
        checks.append(ValidationCheck("1_plan_exists", ok, "plan present" if ok else "no plan / malformed plan"))
        if not ok:
            return fail("no plan to validate")

        unknown = sorted({a for a in plan if universe.get(a) is None})
        checks.append(ValidationCheck("2_actions_known", not unknown,
                                      "all action ids are in the intervention universe" if not unknown
                                      else f"unknown action ids: {unknown}"))
        if unknown:
            return fail(f"plan contains actions outside the configured universe: {unknown}")

        legality = check_plan(system, universe, list(plan), problem.initial_state)
        checks.append(ValidationCheck("3_combination_legal", legality.legal,
                                      "no duplicates, preconditions hold, no forbidden combination"
                                      if legality.legal else f"{legality.code}: {legality.detail}"))
        if not legality.legal:
            return fail(f"illegal plan: {legality.code}: {legality.detail}")

        post = transform(system, problem.initial_state, legality.up, legality.down)
        checks.append(ValidationCheck("4_intervention_applied", True,
                                      f"T_B sets UP {sorted(legality.up)}, DOWN {sorted(legality.down)}"))

        cascade = simulate(system, post)
        checks.append(ValidationCheck("5_cascade_rerun", cascade.fixed_point,
                                      f"{cascade.termination_reason} after {cascade.state_changing_rounds} "
                                      f"state-changing round(s)"))
        if not cascade.fixed_point:
            return fail("cascade did not reach a fixed point")

        obj = compute_objective(system, universe, sorted(plan), cascade.final_state)
        feasible = obj.critical_up >= obj.critical_required
        checks.append(ValidationCheck("6_critical_services", feasible,
                                      f"C(B)={obj.critical_up}, C_min={obj.critical_required}"
                                      + ("" if feasible else f"; critical DOWN: {list(obj.critical_down)}")))

        mismatches: List[str] = []
        rep_cost = _as_fraction(result.cost)
        c7 = rep_cost is not None and rep_cost == obj.cost
        checks.append(ValidationCheck("7_cost", c7, f"recomputed K={cost_to_json(obj.cost)}, reported={result.cost}"))
        if not c7:
            mismatches.append("cost")
        c8 = result.intervention_count == obj.count
        checks.append(ValidationCheck("8_intervention_count", c8,
                                      f"recomputed N={obj.count}, reported={result.intervention_count}"))
        if not c8:
            mismatches.append("intervention_count")
        c9 = result.residual_failures == obj.residual
        checks.append(ValidationCheck("9_residual_failures", c9,
                                      f"recomputed R={obj.residual}, reported={result.residual_failures}"))
        if not c9:
            mismatches.append("residual_failures")

        c10_parts = []
        if result.critical_services_preserved != obj.critical_up:
            c10_parts.append(f"critical preserved reported {result.critical_services_preserved} != {obj.critical_up}")
        if result.objective_value is not None:
            rep = tuple(result.objective_value)
            if len(rep) != 3 or _as_fraction(rep[0]) != obj.cost or rep[1] != obj.count or rep[2] != obj.residual:
                shown = [cost_to_json(v) if isinstance(v, Fraction) else v for v in rep]
                c10_parts.append(f"objective tuple reported {shown} != "
                                 f"[{cost_to_json(obj.cost)}, {obj.count}, {obj.residual}]")
        if result.final_state is not None:
            try:
                reported_state = system.state_from_mapping(result.final_state)
            except (KeyError, TypeError, ValueError):
                reported_state = None
            if reported_state != cascade.final_state:
                c10_parts.append("reported final state differs from the recomputed fixed point")
        checks.append(ValidationCheck("10_reported_vs_recomputed", not c10_parts and not mismatches,
                                      "all reported values match" if not c10_parts and not mismatches
                                      else "; ".join(c10_parts + [f"{m} mismatch" for m in mismatches])))

        base = ValidationResult(False, "", checks, recomputed_objective=obj, recomputed_state=cascade.final_state)
        if not feasible:
            base.reason = "plan does not preserve every designated critical service"
            return base
        if mismatches or c10_parts:
            base.reason = "solver-reported values differ from independent recomputation: " + \
                          "; ".join(c10_parts + [f"{m} mismatch" for m in mismatches])
            return base
        base.valid, base.reason = True, "independently recomputed; all checks passed"
        return base

    # ------------------------------------------------------------------------
    def infeasibility_certificate(self, problem: ContainmentProblem) -> Dict[str, Any]:
        p = problem
        applicable = statically_applicable(p.system, p.actions, p.initial_state)
        state = optimistic_state(p.system, p.actions, p.initial_state, (), applicable)
        final = simulate(p.system, state).final_state
        down = p.system.down_services(final)
        crit_down = [s for s in down if s in p.system.critical]
        return {"certified": bool(crit_down), "optimistic_critical_down": crit_down,
                "optimistic_state_up_effects_from": list(applicable)}

    def validate_infeasibility(self, problem: ContainmentProblem, result: SolverResult,
                               solver_kind: SolverKind) -> ValidationResult:
        checks: List[ValidationCheck] = []
        cert = self.infeasibility_certificate(problem)
        if cert["certified"]:
            checks.append(ValidationCheck("infeasibility_certificate", True,
                                          "even applying every possible UP effect leaves critical services DOWN: "
                                          f"{cert['optimistic_critical_down']} (monotone bound)"))
            return ValidationResult(True, "infeasibility independently certified", checks,
                                    infeasibility_certified=True)
        checks.append(ValidationCheck("infeasibility_certificate", False,
                                      "the monotone bound cannot certify infeasibility (combinations may be illegal)"))
        exact_complete = solver_kind is SolverKind.EXACT and result.search_complete
        checks.append(ValidationCheck("exact_complete_search", exact_complete,
                                      f"solver kind={solver_kind.value}, search_complete={result.search_complete}"))
        if exact_complete and self.accept_uncertified_infeasibility:
            return ValidationResult(True, "infeasibility accepted from a completed exact search (not independently "
                                          "certified; policy allows it)", checks, infeasibility_certified=False)
        return ValidationResult(False, "infeasibility claim could not be verified", checks,
                                infeasibility_certified=False)
