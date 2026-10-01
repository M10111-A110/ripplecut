"""Execution Guard (master spec §32).

Every solver call runs behind this guard, which converts any failure into a
normalized ``SolverResult`` so that a failing solver can never crash RippleCut:

* incompatible problem                  -> INCOMPATIBLE (solver not started)
* exception inside the solver           -> ERROR
* SolverTimeout raised by the context   -> TIMEOUT
* budget / MemoryError / RecursionError -> RESOURCE_LIMIT
* no return within timeout + grace      -> TIMEOUT (see limitation below)
* result of the wrong type / schema     -> INVALID_RESULT

Limitation (documented, not hidden): Python threads cannot be killed. A solver
that ignores its context's cancellation keeps running in a daemon thread after
the guard gives up; its eventual result is discarded and never reaches the
validator. All built-in solvers check the deadline cooperatively.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, Optional

from ..errors import ResourceLimitExceeded, SolverTimeout
from ..model.schema import ContainmentProblem, ResourceLimits
from .base import OptimalityStatus, Solver, SolverContext, SolverResult, SolverStatus


def _fail(name: str, status: SolverStatus, error: str, runtime: float, **meta: Any) -> SolverResult:
    return SolverResult(solver_name=name, status=status, runtime_seconds=runtime,
                        optimality_status=OptimalityStatus.UNKNOWN, error=error, metadata=meta)


def schema_problems(result: SolverResult, expected_name: str) -> Optional[str]:
    """Result schema validation (first step of the validation pipeline, master spec §56)."""
    if not isinstance(result, SolverResult):
        return f"solver returned {type(result).__name__}, not SolverResult"
    if result.solver_name != expected_name:
        return f"result names solver '{result.solver_name}', expected '{expected_name}'"
    if not isinstance(result.status, SolverStatus):
        return f"unknown status {result.status!r}"
    if not isinstance(result.optimality_status, OptimalityStatus):
        return f"unknown optimality status {result.optimality_status!r}"
    plan = result.selected_plan
    if plan is not None and (not isinstance(plan, (tuple, list)) or not all(isinstance(a, str) for a in plan)):
        return "selected_plan must be a sequence of action-id strings"
    if result.status is SolverStatus.SUCCESS:
        if plan is None:
            return "SUCCESS without a selected_plan"
        missing = [f for f in ("cost", "intervention_count", "residual_failures", "critical_services_preserved")
                   if getattr(result, f) is None]
        if missing:
            return f"SUCCESS result is missing reported metrics {missing}"
    if result.status is SolverStatus.NO_FEASIBLE_PLAN and plan is not None:
        return "NO_FEASIBLE_PLAN result must not carry a plan"
    return None


class ExecutionGuard:
    def run(self, solver: Solver, problem: ContainmentProblem, limits: Optional[ResourceLimits] = None) -> SolverResult:
        limits = limits or problem.resource_limits
        name = getattr(solver, "name", "<unnamed>")
        t0 = time.monotonic()
        try:
            reason = solver.supports(problem)
        except Exception as exc:  # a crashing capability check is still just a failed solver
            return _fail(name, SolverStatus.ERROR, f"supports() raised {type(exc).__name__}: {exc}", 0.0)
        if reason:
            return _fail(name, SolverStatus.INCOMPATIBLE, reason, 0.0)

        cancel = threading.Event()
        context = SolverContext(problem, deadline=t0 + limits.timeout_seconds,
                                max_evaluations=limits.max_evaluations, cancel_event=cancel)
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = solver.solve(problem, context)
            except BaseException as exc:  # noqa: BLE001 - isolate everything, including SystemExit
                box["exc"] = exc
                box["tb"] = traceback.format_exc(limit=6)

        thread = threading.Thread(target=target, name=f"solver-{name}", daemon=True)
        thread.start()
        thread.join(limits.timeout_seconds + limits.grace_seconds)
        if thread.is_alive():
            cancel.set()
            thread.join(limits.grace_seconds)
        runtime = time.monotonic() - t0
        stats = context.stats()

        if thread.is_alive():
            return _fail(name, SolverStatus.TIMEOUT,
                         f"no result within {limits.timeout_seconds:g}s (+{limits.grace_seconds:g}s grace); "
                         f"solver ignored cancellation, thread abandoned and its result will be discarded",
                         runtime, abandoned_thread=True, **stats)
        if "exc" in box:
            exc = box["exc"]
            if isinstance(exc, SolverTimeout):
                return _fail(name, SolverStatus.TIMEOUT, str(exc), runtime, **stats)
            if isinstance(exc, (ResourceLimitExceeded, MemoryError, RecursionError)):
                return _fail(name, SolverStatus.RESOURCE_LIMIT, f"{type(exc).__name__}: {exc}", runtime, **stats)
            return _fail(name, SolverStatus.ERROR, f"{type(exc).__name__}: {exc}", runtime,
                         traceback=box.get("tb", ""), **stats)

        result = box.get("result")
        problem_text = schema_problems(result, name)
        if problem_text:
            return _fail(name, SolverStatus.INVALID_RESULT, f"malformed result: {problem_text}", runtime, **stats)
        # The guard's wall-clock measurement is authoritative; solver-reported runtime is ignored.
        return replace(result, runtime_seconds=runtime,
                       metadata={**dict(result.metadata), "guard_runtime_seconds": round(runtime, 6)})
