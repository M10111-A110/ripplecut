"""The common Solver Interface (master spec §25-§27).

A solver *searches*; RippleCut *simulates, evaluates and verifies*.

Solvers receive exactly two things: the immutable ``ContainmentProblem`` and a
``SolverContext``. The context is the only way a solver can evaluate a plan or
an optimistic bound, and it routes every call through RippleCut's single
intervention engine, cascade simulator and objective evaluator. Solvers
therefore cannot own or redefine dependency semantics, the objective or
validation rules (master spec §98, §99, §122).
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..engine.interventions import optimistic_state, statically_applicable
from ..engine.objective import ObjectiveKey, PlanEvaluation, evaluate_plan, evaluate_state_bound, objective_key
from ..errors import ResourceLimitExceeded, SolverTimeout
from ..model.schema import ContainmentProblem, Plan, cost_to_json


class SolverStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INCOMPATIBLE = "INCOMPATIBLE"
    ERROR = "ERROR"
    INVALID_RESULT = "INVALID_RESULT"


class OptimalityStatus(str, Enum):
    PROVEN_OPTIMAL = "PROVEN_OPTIMAL"
    VALID_NOT_PROVEN_OPTIMAL = "VALID_NOT_PROVEN_OPTIMAL"
    HEURISTIC = "HEURISTIC"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"     # final outcome NO_FEASIBLE_PLAN: there is no plan to rank


class SolverKind(str, Enum):
    EXACT = "EXACT"            # completes => lexicographic optimum over the finite model
    HEURISTIC = "HEURISTIC"    # never claims optimality


@dataclass(frozen=True)
class SolverCapabilities:
    kind: SolverKind
    max_actions: Optional[int] = None      # None = no declared limit
    description: str = ""


@dataclass(frozen=True)
class SolverResult:
    """Standardized solver output (master spec §27). Metrics are the solver's *claims*."""

    solver_name: str
    status: SolverStatus
    selected_plan: Optional[Tuple[str, ...]] = None
    objective_value: Optional[Tuple[Any, ...]] = None       # claimed (K, N, R)
    cost: Optional[Any] = None                              # claimed K(B)
    intervention_count: Optional[int] = None                # claimed N(B)
    residual_failures: Optional[int] = None                 # claimed R(B)
    critical_services_preserved: Optional[int] = None       # claimed C(B)
    final_state: Optional[Mapping[str, int]] = None         # claimed x*_B (optional)
    runtime_seconds: float = 0.0
    nodes_explored: Optional[int] = None
    search_complete: bool = False
    optimality_status: OptimalityStatus = OptimalityStatus.UNKNOWN   # solver's claim; NOT trusted
    validation_status: str = "NOT_VALIDATED"
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_name": self.solver_name, "status": self.status.value,
            "selected_plan": list(self.selected_plan) if self.selected_plan is not None else None,
            "objective_value": [cost_to_json(self.objective_value[0]), *self.objective_value[1:]]
            if self.objective_value else None,
            "cost": cost_to_json(self.cost) if isinstance(self.cost, (int, Fraction)) else self.cost,
            "intervention_count": self.intervention_count, "residual_failures": self.residual_failures,
            "critical_services_preserved": self.critical_services_preserved,
            "runtime_seconds": round(self.runtime_seconds, 6), "nodes_explored": self.nodes_explored,
            "search_complete": self.search_complete, "optimality_claim": self.optimality_status.value,
            "validation_status": self.validation_status, "error": self.error,
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return cost_to_json(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass(frozen=True)
class Bound:
    critical_upper: int      # UB_C: no legal plan in the subtree preserves more critical services
    residual_lower: int      # LB_R: no legal plan in the subtree has fewer residual failures


class SolverContext:
    """Per-run access to RippleCut's engine plus execution limits.

    Created by the execution guard for each solver invocation. Every evaluation
    checks the deadline, the cancellation flag and the evaluation budget, and
    raises ``SolverTimeout`` / ``ResourceLimitExceeded`` when one is exceeded.
    """

    def __init__(self, problem: ContainmentProblem, deadline: float, max_evaluations: int,
                 cancel_event: Optional[threading.Event] = None) -> None:
        self._problem = problem
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.cancel_event = cancel_event or threading.Event()
        self.candidates_examined = 0
        self.plan_simulations = 0
        self.bound_simulations = 0
        self.started = time.monotonic()

    # ---- limits -------------------------------------------------------------
    @property
    def simulations(self) -> int:
        return self.plan_simulations + self.bound_simulations

    def check_limits(self) -> None:
        if self.cancel_event.is_set() or time.monotonic() > self.deadline:
            raise SolverTimeout("solver deadline exceeded")
        if self.simulations >= self.max_evaluations:
            raise ResourceLimitExceeded(f"evaluation budget of {self.max_evaluations} cascade simulations exhausted")

    def time_remaining(self) -> float:
        return self.deadline - time.monotonic()

    # ---- engine access (single source of truth) ------------------------------
    def evaluate(self, plan: Sequence[str]) -> PlanEvaluation:
        self.check_limits()
        self.candidates_examined += 1
        ev = evaluate_plan(self._problem, plan)
        if ev.simulated:
            self.plan_simulations += 1
        return ev

    def optimistic_bound(self, included: Sequence[str], candidates: Sequence[str]) -> Bound:
        self.check_limits()
        p = self._problem
        state = optimistic_state(p.system, p.actions, p.initial_state, included, candidates)
        self.bound_simulations += 1
        c_up, r = evaluate_state_bound(p, state)
        return Bound(critical_upper=c_up, residual_lower=r)

    def applicable_actions(self) -> Tuple[str, ...]:
        p = self._problem
        return statically_applicable(p.system, p.actions, p.initial_state)

    @staticmethod
    def key(ev: PlanEvaluation) -> ObjectiveKey:
        return ev.key

    def stats(self) -> Dict[str, int]:
        return {"candidates_examined": self.candidates_examined, "plan_simulations": self.plan_simulations,
                "bound_simulations": self.bound_simulations}


class Solver(ABC):
    """Implement ``name``, ``capabilities`` and ``solve``; register it; done (master spec §28)."""

    name: str = ""
    capabilities: SolverCapabilities = SolverCapabilities(kind=SolverKind.HEURISTIC)

    def supports(self, problem: ContainmentProblem) -> Optional[str]:
        """Return None if the solver can handle the problem, else a human-readable reason."""
        limit = self.capabilities.max_actions
        if limit is not None and problem.m > limit:
            return f"{self.name} declares max_actions={limit}, problem has m={problem.m}"
        return None

    @abstractmethod
    def solve(self, problem: ContainmentProblem, context: SolverContext) -> SolverResult:
        ...


def result_from_evaluation(solver_name: str, status: SolverStatus, ev: PlanEvaluation, *,
                           search_complete: bool, optimality: OptimalityStatus,
                           nodes_explored: Optional[int], metadata: Mapping[str, Any],
                           system_ids: Optional[Tuple[str, ...]] = None) -> SolverResult:
    """Package a canonical PlanEvaluation as the solver's reported result."""
    obj = ev.objective
    assert obj is not None and ev.cascade is not None
    final_state = None
    if system_ids is not None:
        final_state = {sid: int(ev.cascade.final_state[i]) for i, sid in enumerate(system_ids)}
    return SolverResult(solver_name=solver_name, status=status, selected_plan=tuple(ev.plan),
                        objective_value=(obj.cost, obj.count, obj.residual), cost=obj.cost,
                        intervention_count=obj.count, residual_failures=obj.residual,
                        critical_services_preserved=obj.critical_up, final_state=final_state,
                        nodes_explored=nodes_explored, search_complete=search_complete,
                        optimality_status=optimality, metadata=dict(metadata))
