"""The single RippleCut objective evaluator (master spec §23, PDF §8).

For a legal candidate plan B with post-cascade state x*_B = Cascade(T_B(x(0))):

    C(B) = sum_v c(v) x*_v(B)          critical services UP
    K(B) = sum_{a in B} k(a)           total configured cost   (exact rationals)
    N(B) = |B|                         number of interventions
    R(B) = sum_v 1[x*_v(B) = 0]        residual failed services (all services)

    F_feas = { B legal : C(B) >= C_min }
    B*     = lexmin over F_feas of (K(B), N(B), R(B)), then sorted action ids.

The final tie-breaker only applies when (K, N, R) are all equal, which makes
the order total and the selected plan reproducible. No weighted score exists
anywhere in the decision path.

``evaluate_plan`` is the canonical evaluation used by every solver (through its
SolverContext) and, independently, by the safety validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Optional, Sequence, Tuple

from ..model.schema import ActionUniverse, ContainmentProblem, Plan, State, SystemModel, cost_to_json
from .interventions import LegalityResult, check_plan, transform
from .simulator import CascadeResult, cascade_fixed_point, simulate

ObjectiveKey = Tuple[Fraction, int, int, Plan]


@dataclass(frozen=True)
class ObjectiveValue:
    critical_up: int            # C(B)
    critical_required: int      # C_min
    cost: Fraction              # K(B)
    count: int                  # N(B)
    residual: int               # R(B)
    residual_services: Tuple[str, ...]
    critical_down: Tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return self.critical_up >= self.critical_required

    def tuple(self) -> Tuple[Fraction, int, int]:
        return (self.cost, self.count, self.residual)

    def to_dict(self) -> Dict[str, Any]:
        return {"feasible": self.feasible, "C": self.critical_up, "C_min": self.critical_required,
                "K": cost_to_json(self.cost), "N": self.count, "R": self.residual,
                "residual_services": list(self.residual_services), "critical_down": list(self.critical_down)}


@dataclass(frozen=True)
class PlanEvaluation:
    plan: Plan
    legality: LegalityResult
    post_intervention_state: Optional[State]
    cascade: Optional[CascadeResult]
    objective: Optional[ObjectiveValue]

    @property
    def legal(self) -> bool:
        return self.legality.legal

    @property
    def simulated(self) -> bool:
        return self.cascade is not None

    @property
    def feasible(self) -> bool:
        return self.legal and self.objective is not None and self.objective.feasible

    @property
    def key(self) -> ObjectiveKey:
        if self.objective is None:
            raise ValueError("an illegal or unsimulated plan has no objective key")
        return objective_key(self.objective, self.plan)


def plan_cost(universe: ActionUniverse, plan: Sequence[str]) -> Fraction:
    return sum((universe.get(a).cost for a in plan), Fraction(0))


def compute_objective(system: SystemModel, universe: ActionUniverse, plan: Sequence[str],
                      final_state: State) -> ObjectiveValue:
    down = system.down_services(final_state)
    critical_down = tuple(s for s in down if s in system.critical)
    return ObjectiveValue(critical_up=len(system.critical) - len(critical_down),
                          critical_required=system.c_min,
                          cost=plan_cost(universe, plan), count=len(plan),
                          residual=len(down), residual_services=down, critical_down=critical_down)


def objective_key(value: ObjectiveValue, plan: Sequence[str]) -> ObjectiveKey:
    """Total order used by EVERY solver: (K, N, R, sorted action ids)."""
    return (value.cost, value.count, value.residual, tuple(sorted(plan)))


def evaluate_plan(problem: ContainmentProblem, plan: Sequence[str]) -> PlanEvaluation:
    """Legality -> T_B -> cascade -> metrics. Illegal plans are rejected before any simulation."""
    system, universe = problem.system, problem.actions
    canon = tuple(sorted(plan)) if all(isinstance(a, str) for a in plan) else tuple(plan)
    legality = check_plan(system, universe, canon, problem.initial_state)
    if not legality.legal:
        return PlanEvaluation(canon, legality, None, None, None)
    post = transform(system, problem.initial_state, legality.up, legality.down)
    cascade = simulate(system, post)
    if not cascade.fixed_point:
        return PlanEvaluation(canon, legality, post, cascade, None)
    return PlanEvaluation(canon, legality, post, cascade,
                          compute_objective(system, universe, canon, cascade.final_state))


def evaluate_state_bound(problem: ContainmentProblem, state: State) -> Tuple[int, int]:
    """(C, R) of Cascade(state); used for optimistic bounds (monotone in ``state``)."""
    final = cascade_fixed_point(problem.system, state)      # raises instead of returning a non-fixed point
    down = problem.system.down_services(final)
    crit_up = len(problem.system.critical) - sum(1 for s in down if s in problem.system.critical)
    return crit_up, len(down)
