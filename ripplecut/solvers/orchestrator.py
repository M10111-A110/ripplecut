"""Solver Orchestrator (master spec §29, §33, §57-§61).

Workflow (identical for 1, 2, 30 or N registered solvers):

    for solver in policy.execution_order():            # configuration, not code
        result = guard.run(solver)                      # crash/timeout/malformed -> normalized
        if result proposes a plan or infeasibility:
            validation = validator.validate(result)     # independent recomputation
            if valid: accept and stop
        record failure -> next solver (fallback)
    none accepted -> NO_VALID_SOLVER_RESULT (never a fabricated plan)

Optimality is derived by RippleCut, not taken from the solver's claim:
    EXACT solver, search complete, validated   -> PROVEN_OPTIMAL
    HEURISTIC solver, validated                  -> HEURISTIC
    validated incumbent of a timed-out solver    -> VALID_NOT_PROVEN_OPTIMAL
A HEURISTIC result can become PROVEN_OPTIMAL only through the verification
layer: an EXACT cross-check solver that completes and whose independently
validated objective (K, N, R) is identical. A cross-check mismatch downgrades
PROVEN_OPTIMAL to VALID_NOT_PROVEN_OPTIMAL. A NO_FEASIBLE_PLAN claim accepted
without an independent certificate is replaced if a cross-check solver returns
a validated feasible plan (a constructive refutation); a valid but suboptimal
plan is never silently replaced, only flagged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from ..engine.objective import objective_key
from ..logs import get_logger, log_event
from ..model.schema import ContainmentProblem, cost_to_json
from ..validation.validator import SafetyValidator, ValidationResult
from .base import OptimalityStatus, SolverKind, SolverResult, SolverStatus
from .guard import ExecutionGuard
from .policy import SolverPolicy
from .registry import SolverRegistry

FINAL_SUCCESS = "SUCCESS"
FINAL_NO_FEASIBLE = "NO_FEASIBLE_PLAN"
FINAL_NO_VALID = "NO_VALID_SOLVER_RESULT"


@dataclass
class SolverAttempt:
    solver_name: str
    kind: str
    result: SolverResult
    outcome: str                     # ACCEPTED | REJECTED_BY_VALIDATOR | FAILED | INCUMBENT_HELD | NOT_REGISTERED
    validation: Optional[ValidationResult] = None

    def to_dict(self, problem: ContainmentProblem) -> Dict[str, Any]:
        return {"solver": self.solver_name, "kind": self.kind, "outcome": self.outcome,
                "result": self.result.to_dict(),
                "validation": self.validation.to_dict(problem) if self.validation else None}


@dataclass
class OrchestrationResult:
    status: str
    attempts: List[SolverAttempt]
    selected_solver: Optional[str] = None
    fallback_count: int = 0
    plan: Optional[tuple] = None
    validation: Optional[ValidationResult] = None
    optimality_status: OptimalityStatus = OptimalityStatus.UNKNOWN
    optimality_basis: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def accepted_result(self) -> Optional[SolverResult]:
        for a in self.attempts:
            if a.outcome == "ACCEPTED":
                return a.result
        return None

    def to_dict(self, problem: ContainmentProblem) -> Dict[str, Any]:
        obj = self.validation.recomputed_objective if self.validation else None
        return {"status": self.status, "selected_solver": self.selected_solver,
                "fallback_count": self.fallback_count,
                "plan": list(self.plan) if self.plan is not None else None,
                "objective": obj.to_dict() if obj else None,
                "optimality_status": self.optimality_status.value, "optimality_basis": self.optimality_basis,
                "validation": self.validation.to_dict(problem) if self.validation else None,
                "attempts": [a.to_dict(problem) for a in self.attempts],
                "verification": self.verification, "diagnostics": list(self.diagnostics)}


class SolverOrchestrator:
    def __init__(self, registry: SolverRegistry, policy: SolverPolicy,
                 validator: Optional[SafetyValidator] = None, guard: Optional[ExecutionGuard] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        policy.validate_against(registry)
        self.registry = registry
        self.policy = policy
        self.validator = validator or SafetyValidator(policy.accept_uncertified_infeasibility_from_exact_solver)
        self.guard = guard or ExecutionGuard()
        self.log = logger or get_logger()

    # ------------------------------------------------------------------------
    def solve(self, problem: ContainmentProblem) -> OrchestrationResult:
        attempts: List[SolverAttempt] = []
        held: Optional[SolverAttempt] = None          # validated incumbent of a timed-out solver
        order = self.policy.execution_order()
        for position, name in enumerate(order):
            solver = self.registry.maybe_get(name)
            if solver is None:                         # defensive; policy was validated at construction
                attempts.append(SolverAttempt(name, "UNKNOWN", SolverResult(name, SolverStatus.ERROR,
                                              error="not registered"), "NOT_REGISTERED"))
                continue
            kind = solver.capabilities.kind
            result = self.guard.run(solver, problem, self.policy.limits)
            log_event(self.log, "solver_attempt", solver=name, position=position, status=result.status.value,
                      runtime_seconds=round(result.runtime_seconds, 6), error=result.error)

            if result.status in (SolverStatus.SUCCESS, SolverStatus.NO_FEASIBLE_PLAN):
                validation = self.validator.validate(problem, result, kind)
                log_event(self.log, "validation", solver=name, valid=validation.valid, reason=validation.reason,
                          checks={c.name: c.passed for c in validation.checks})
                if validation.valid:
                    attempt = SolverAttempt(name, kind.value, replace(result, validation_status="VALID"),
                                            "ACCEPTED", validation)
                    attempts.append(attempt)
                    return self._finalize(problem, attempts, attempt, position)
                attempts.append(SolverAttempt(name, kind.value, replace(
                    result, status=SolverStatus.INVALID_RESULT, validation_status="INVALID",
                    error=f"rejected by independent validator: {validation.reason}",
                    metadata={**dict(result.metadata), "original_status": result.status.value}),
                    "REJECTED_BY_VALIDATOR", validation))
                log_event(self.log, "fallback", from_solver=name, reason="INVALID_RESULT")
                continue

            if (result.status in (SolverStatus.TIMEOUT, SolverStatus.RESOURCE_LIMIT)
                    and result.selected_plan is not None and held is None):
                validation = self.validator.validate_plan(problem, result)
                if validation.valid:
                    held = SolverAttempt(name, kind.value, replace(result, validation_status="VALID"),
                                         "INCUMBENT_HELD", validation)
                    attempts.append(held)
                    log_event(self.log, "fallback", from_solver=name, reason=result.status.value,
                              incumbent_held=True)
                    continue
                attempts.append(SolverAttempt(name, kind.value, result, "FAILED", validation))
                log_event(self.log, "fallback", from_solver=name, reason=result.status.value)
                continue

            attempts.append(SolverAttempt(name, kind.value, result, "FAILED"))
            log_event(self.log, "fallback", from_solver=name, reason=result.status.value)

        if held is not None and self.policy.accept_validated_incumbent_if_all_fail:
            held.outcome = "ACCEPTED"
            return self._finalize(problem, attempts, held, attempts.index(held), incumbent=True)

        out = OrchestrationResult(status=FINAL_NO_VALID, attempts=attempts, fallback_count=max(0, len(attempts) - 1),
                                  optimality_status=OptimalityStatus.UNKNOWN,
                                  diagnostics=[f"{a.solver_name}: {a.result.status.value}: {a.result.error}"
                                               for a in attempts])
        if held is not None:
            out.diagnostics.append("a validated incumbent exists but policy "
                                   "'accept_validated_incumbent_if_all_fail' is false, so it is not recommended")
        log_event(self.log, "no_valid_solver_result", attempted=[a.solver_name for a in attempts])
        return out

    # ------------------------------------------------------------------------
    def _finalize(self, problem: ContainmentProblem, attempts: List[SolverAttempt], accepted: SolverAttempt,
                  position: int, incumbent: bool = False) -> OrchestrationResult:
        res, kind = accepted.result, accepted.kind
        diagnostics: List[str] = []
        if incumbent:
            optimality, basis = OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL, \
                f"validated incumbent of {res.solver_name} after {res.status.value}; search incomplete"
        elif kind == SolverKind.EXACT.value and res.search_complete:
            optimality, basis = OptimalityStatus.PROVEN_OPTIMAL, \
                f"exact solver {res.solver_name} completed its search; result independently validated"
        elif kind == SolverKind.HEURISTIC.value:
            optimality, basis = OptimalityStatus.HEURISTIC, f"heuristic solver {res.solver_name}; plan validated"
        else:
            optimality, basis = OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL, "validated; optimality not established"
        claim = res.optimality_status
        if claim is OptimalityStatus.PROVEN_OPTIMAL and optimality is not OptimalityStatus.PROVEN_OPTIMAL:
            diagnostics.append(f"{res.solver_name} claimed PROVEN_OPTIMAL; claim not accepted ({basis})")

        status = FINAL_NO_FEASIBLE if res.status is SolverStatus.NO_FEASIBLE_PLAN else FINAL_SUCCESS
        if status == FINAL_NO_FEASIBLE:
            certified = bool(accepted.validation and accepted.validation.infeasibility_certified)
            optimality = OptimalityStatus.NOT_APPLICABLE
            basis = ("infeasibility independently CERTIFIED by the validator's monotone bound" if certified else
                     f"infeasibility NOT independently certified; accepted from the completed exact search of "
                     f"{res.solver_name} under policy")
        out = OrchestrationResult(status=status, attempts=attempts, selected_solver=res.solver_name,
                                  fallback_count=position, plan=res.selected_plan if status == FINAL_SUCCESS else None,
                                  validation=accepted.validation, optimality_status=optimality,
                                  optimality_basis=basis, diagnostics=diagnostics)
        out.verification = self._cross_check(problem, out)
        log_event(self.log, "final_plan", status=status, solver=res.solver_name, fallback_count=position,
                  plan=list(out.plan) if out.plan is not None else None,
                  optimality=out.optimality_status.value, verification=out.verification.get("status"))
        return out

    # ---- verification layer --------------------------------------------------
    def _cross_check(self, problem: ContainmentProblem, out: OrchestrationResult) -> Dict[str, Any]:
        if not self.policy.cross_check_solvers:
            return {"status": "NOT_CONFIGURED", "checks": []}
        names = [n for n in self.policy.cross_check_solvers if n != out.selected_solver]
        if not names:
            return {"status": "NOT_NEEDED", "reason": f"the accepted result was produced by the configured "
                                                      f"cross-check solver {out.selected_solver}", "checks": []}
        if problem.m > self.policy.cross_check_max_actions:
            return {"status": "SKIPPED", "reason": f"m={problem.m} exceeds cross_check max_actions="
                                                   f"{self.policy.cross_check_max_actions}", "checks": []}
        accepted_key = None
        if out.status == FINAL_SUCCESS and out.validation and out.validation.recomputed_objective:
            accepted_key = objective_key(out.validation.recomputed_objective, out.plan or ())
        checks = []
        overall = "EQUIVALENT"
        refutation = None
        for name in names:
            solver = self.registry.get(name)
            res = self.guard.run(solver, problem, self.policy.limits)
            entry: Dict[str, Any] = {"solver": name, "status": res.status.value,
                                     "runtime_seconds": round(res.runtime_seconds, 6),
                                     "metadata": {k: v for k, v in res.metadata.items()
                                                  if k in ("candidates_total", "legal", "feasible", "illegal",
                                                           "plan_simulations", "top_feasible")}}
            if res.status not in (SolverStatus.SUCCESS, SolverStatus.NO_FEASIBLE_PLAN):
                entry["verdict"] = "CROSS_CHECK_FAILED"
                overall = "INCONCLUSIVE" if overall == "EQUIVALENT" else overall
                checks.append(entry)
                continue
            val = self.validator.validate(problem, res, solver.capabilities.kind)
            if not val.valid:
                entry["verdict"] = "CROSS_CHECK_RESULT_INVALID"
                overall = "INCONCLUSIVE" if overall == "EQUIVALENT" else overall
                checks.append(entry)
                continue
            if res.status is SolverStatus.NO_FEASIBLE_PLAN or out.status == FINAL_NO_FEASIBLE:
                same = res.status is SolverStatus.NO_FEASIBLE_PLAN and out.status == FINAL_NO_FEASIBLE
                entry["verdict"] = "EQUIVALENT" if same else "MISMATCH"
                if not same and res.status is SolverStatus.SUCCESS and refutation is None:
                    # A validated feasible plan is a constructive counterexample to the accepted
                    # (uncertified) infeasibility claim: that claim was wrong, not merely suboptimal.
                    entry["verdict"] = "REFUTES_INFEASIBILITY"
                    entry["plan"] = list(res.selected_plan or ())
                    refutation = (name, solver, res, val)
            else:
                key = objective_key(val.recomputed_objective, res.selected_plan or ())
                entry["plan"] = list(res.selected_plan or ())
                entry["objective"] = [cost_to_json(key[0]), key[1], key[2]]
                if key == accepted_key:
                    entry["verdict"] = "EQUIVALENT"
                elif accepted_key is not None and key[:3] == accepted_key[:3]:
                    entry["verdict"] = "EQUIVALENT_OBJECTIVE_DIFFERENT_TIE_BREAK"
                else:
                    entry["verdict"] = "MISMATCH"
                    entry["better_than_accepted"] = accepted_key is not None and key < accepted_key
            if entry["verdict"].startswith("EQUIVALENT"):
                exact_complete = solver.capabilities.kind is SolverKind.EXACT and res.search_complete
                if exact_complete and out.optimality_status in (OptimalityStatus.HEURISTIC,
                                                                OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL):
                    out.optimality_status = OptimalityStatus.PROVEN_OPTIMAL
                    out.optimality_basis += f"; upgraded: exact cross-check by {name} found the identical optimum"
            else:
                overall = "MISMATCH"
                if out.optimality_status is OptimalityStatus.PROVEN_OPTIMAL:
                    out.optimality_status = OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL
                    out.optimality_basis += f"; DOWNGRADED: cross-check by {name} disagrees"
                out.diagnostics.append(f"verification mismatch with {name}: {entry}")
            checks.append(entry)
        if refutation is not None:
            self._apply_refutation(problem, out, *refutation)
        return {"status": overall, "checks": checks}

    def _apply_refutation(self, problem: ContainmentProblem, out: OrchestrationResult, name: str, solver: Any,
                          res: SolverResult, val: ValidationResult) -> None:
        for a in out.attempts:
            if a.outcome == "ACCEPTED":
                a.outcome = "REFUTED_BY_CROSS_CHECK"
        exact_complete = solver.capabilities.kind is SolverKind.EXACT and res.search_complete
        refuted = out.selected_solver
        out.attempts.append(SolverAttempt(name, solver.capabilities.kind.value, replace(res, validation_status="VALID"),
                                          "ACCEPTED", val))
        out.status, out.plan, out.validation, out.selected_solver = FINAL_SUCCESS, res.selected_plan, val, name
        out.fallback_count = len(out.attempts) - 1
        out.optimality_status = (OptimalityStatus.PROVEN_OPTIMAL if exact_complete
                                 else OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL)
        out.optimality_basis = (f"the uncertified NO_FEASIBLE_PLAN claim of {refuted} was refuted by a validated "
                                f"feasible plan from cross-check solver {name}"
                                + ("; that solver is exact and completed its search" if exact_complete else ""))
        out.diagnostics.append(out.optimality_basis)
        log_event(self.log, "infeasibility_refuted", refuted_solver=refuted, by=name,
                  plan=list(res.selected_plan or ()))
