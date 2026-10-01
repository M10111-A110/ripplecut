"""Solver failure simulation / resilience test (master spec §68, §93, §107).

``FaultInjectingSolver`` wraps a registered solver and deliberately makes it
fail in a chosen way while keeping the wrapped solver's name, so the
orchestrator treats it exactly like the real one. It exists only to
demonstrate and test the fallback architecture; it is never part of a normal
run and every report produced with it is labeled accordingly.

Modes:
  crash           raise an exception                      -> guard: ERROR
  timeout         spin cooperatively until the deadline   -> guard: TIMEOUT
  invalid_action  report a plan with an unknown action    -> validator: INVALID_RESULT
  wrong_cost      run the real solver, then misreport K   -> validator: INVALID_RESULT
  malformed       return a non-SolverResult object        -> guard: INVALID_RESULT
"""
from __future__ import annotations

import time
from dataclasses import replace
from fractions import Fraction

from ..model.schema import ContainmentProblem
from .base import OptimalityStatus, Solver, SolverContext, SolverResult, SolverStatus

FAULT_MODES = ("crash", "timeout", "invalid_action", "wrong_cost", "malformed")
RESILIENCE_LABEL = "Solver failure simulation / resilience test"


class InjectedSolverFault(RuntimeError):
    """Raised on purpose by the 'crash' mode."""


class FaultInjectingSolver(Solver):
    def __init__(self, inner: Solver, mode: str) -> None:
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}; choose from {FAULT_MODES}")
        self.inner = inner
        self.mode = mode
        self.name = inner.name
        self.capabilities = inner.capabilities

    def supports(self, problem: ContainmentProblem):
        return self.inner.supports(problem)

    def solve(self, problem: ContainmentProblem, context: SolverContext) -> SolverResult:
        if self.mode == "crash":
            raise InjectedSolverFault(f"{RESILIENCE_LABEL}: injected crash in solver '{self.name}'")
        if self.mode == "timeout":
            while True:                                  # cooperative: exits via SolverTimeout
                context.check_limits()
                time.sleep(0.01)
        if self.mode == "malformed":
            return {"plan": "not a SolverResult"}        # type: ignore[return-value]
        if self.mode == "invalid_action":
            return SolverResult(self.name, SolverStatus.SUCCESS, selected_plan=("__injected_unknown_action__",),
                                objective_value=(Fraction(0), 1, 0), cost=Fraction(0), intervention_count=1,
                                residual_failures=0, critical_services_preserved=problem.system.c_min,
                                search_complete=True, optimality_status=OptimalityStatus.PROVEN_OPTIMAL,
                                metadata={"fault_injection": self.mode})
        # wrong_cost: a genuine result whose reported cost is corrupted
        real = self.inner.solve(problem, context)
        if real.cost is None:          # e.g. a (correct) NO_FEASIBLE_PLAN result: there is no cost to corrupt
            return replace(real, metadata={**dict(real.metadata),
                                           "fault_injection": "wrong_cost not applicable: result has no cost"})
        bad = Fraction(real.cost) - 1
        return replace(real, cost=bad, objective_value=(bad, *real.objective_value[1:]) if real.objective_value else None,
                       metadata={**dict(real.metadata), "fault_injection": self.mode})
