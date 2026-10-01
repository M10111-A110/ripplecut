"""The single RippleCut cascade simulator (master spec §16-§19, PDF §5).

Transition (synchronous, deterministic):

    Phi(x)_v = x_v AND R_v(x)

The ``x_v AND`` factor encodes the MVP's no-recovery assumption: a service that
is DOWN stays DOWN during the uncontrolled cascade (failures are latched). It
makes the two properties used by the termination proof hold *by construction*
for every configured model, including cyclic ones:

  (P1) Phi(x) <= x                    (no spontaneous recovery)
  (P2) x <= y  =>  Phi(x) <= Phi(y)   (monotonicity; each R_v is monotone)

Starting from an initial state with all non-failed services UP, the trajectory
is identical to iterating x -> R(x) (PDF §5.3), because the sequence is
descending. By (P1) every state-changing round turns at least one 1 into a 0,
so at most n = |V| state-changing rounds occur and one further evaluation
confirms the fixed point x* = Phi(x*). A defensive hard bound of |V|+1 Phi
evaluations is enforced anyway; reaching it returns an explicit
SIMULATION_ERROR instead of looping.

Interventions never happen inside the cascade: a plan B is applied once to
x(0) and then the same Phi is iterated (PDF §7.2: x*_B = Cascade(T_B(x(0)))).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..errors import SimulationError
from ..model.schema import State, SystemModel
from .dependency import compiled_rules

FIXED_POINT = "FIXED_POINT"
ROUND_LIMIT = "ROUND_LIMIT_EXCEEDED"
RECOVERY_VIOLATION = "NO_RECOVERY_ASSUMPTION_VIOLATED"


@dataclass(frozen=True)
class CascadeRound:
    round: int                          # t (state x(t) after t applications of Phi)
    state: State
    newly_failed: Tuple[str, ...]       # services with x(t-1)=1 and x(t)=0


@dataclass(frozen=True)
class CascadeResult:
    initial_state: State
    final_state: State
    history: Tuple[CascadeRound, ...]   # round 0 = initial state
    state_changing_rounds: int          # T such that x(T) = x*
    phi_evaluations: int                # T + 1 (the last one confirms the fixed point)
    round_limit: int
    termination_reason: str
    fixed_point: bool

    def failed_services(self, system: SystemModel) -> Tuple[str, ...]:
        return system.down_services(self.final_state)

    def to_dict(self, system: SystemModel) -> Dict:
        return {
            "initial_down": list(system.down_services(self.initial_state)),
            "rounds": [{"round": r.round, "newly_failed": list(r.newly_failed),
                        "down": list(system.down_services(r.state))} for r in self.history],
            "final_down": list(system.down_services(self.final_state)),
            "final_state": system.state_to_mapping(self.final_state),
            "state_changing_rounds": self.state_changing_rounds,
            "phi_evaluations": self.phi_evaluations,
            "round_limit": self.round_limit,
            "termination_reason": self.termination_reason,
            "fixed_point": self.fixed_point,
        }


def phi(system: SystemModel, x: State) -> State:
    """One synchronous cascade round: Phi(x)_v = x_v AND R_v(x)."""
    nxt = list(x)
    for cr in compiled_rules(system):
        i = cr.downstream_index
        if x[i] == 1 and cr.evaluate(x) == 0:
            nxt[i] = 0
    return tuple(nxt)


def _validate_state(system: SystemModel, x: State) -> None:
    if len(x) != len(system.services):
        raise SimulationError(f"state has {len(x)} entries but the model has {len(system.services)} services")
    if any(v not in (0, 1) for v in x):
        raise SimulationError("formal Boolean state entries must be 0 or 1")


def simulate(system: SystemModel, initial_state: State, max_rounds: Optional[int] = None) -> CascadeResult:
    """Iterate Phi from ``initial_state`` to its fixed point and return the full trace."""
    _validate_state(system, initial_state)
    limit = len(system.services) + 1 if max_rounds is None else int(max_rounds)
    ids = system.service_ids
    history: List[CascadeRound] = [CascadeRound(0, initial_state, ())]
    x = initial_state
    evaluations = 0
    while evaluations < limit:
        nxt = phi(system, x)
        evaluations += 1
        if nxt == x:
            return CascadeResult(initial_state, x, tuple(history), len(history) - 1, evaluations, limit,
                                 FIXED_POINT, True)
        if any(a == 0 and b == 1 for a, b in zip(x, nxt)):     # defensive check of (P1)
            return CascadeResult(initial_state, nxt, tuple(history), len(history) - 1, evaluations, limit,
                                 RECOVERY_VIOLATION, False)
        newly = tuple(ids[i] for i in range(len(ids)) if x[i] == 1 and nxt[i] == 0)
        history.append(CascadeRound(len(history), nxt, newly))
        x = nxt
    return CascadeResult(initial_state, x, tuple(history), len(history) - 1, evaluations, limit,
                         ROUND_LIMIT, False)


def cascade_fixed_point(system: SystemModel, initial_state: State) -> State:
    """Fixed point only; raises SimulationError if it is not reached (never loops)."""
    result = simulate(system, initial_state)
    if not result.fixed_point:
        raise SimulationError(f"cascade did not reach a fixed point: {result.termination_reason}",
                              details={"termination_reason": result.termination_reason})
    return result.final_state
