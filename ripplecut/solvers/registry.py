"""Solver Registry (master spec §28).

Adding an algorithm = implement ``Solver`` + ``register`` (+ optionally list it
in the solver policy). Nothing else in RippleCut changes. The registry holds no
global state: callers build and pass registry instances explicitly.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from ..errors import ErrorCode, RippleCutError
from .base import Solver, SolverCapabilities


class SolverRegistry:
    def __init__(self, solvers: Iterable[Solver] = ()) -> None:
        self._solvers: Dict[str, Solver] = {}
        for s in solvers:
            self.register(s)

    @staticmethod
    def _check(solver: Solver) -> None:
        name = getattr(solver, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise RippleCutError("solver must have a non-empty string 'name'", ErrorCode.CONFIG_ERROR)
        if not callable(getattr(solver, "solve", None)):
            raise RippleCutError(f"solver '{name}' has no callable solve()", ErrorCode.CONFIG_ERROR)
        if not isinstance(getattr(solver, "capabilities", None), SolverCapabilities):
            raise RippleCutError(f"solver '{name}' must declare SolverCapabilities", ErrorCode.CONFIG_ERROR)

    def register(self, solver: Solver, replace: bool = False) -> None:
        self._check(solver)
        if solver.name in self._solvers and not replace:
            raise RippleCutError(f"solver '{solver.name}' is already registered", ErrorCode.CONFIG_ERROR)
        self._solvers[solver.name] = solver

    def unregister(self, name: str) -> None:
        if name not in self._solvers:
            raise RippleCutError(f"solver '{name}' is not registered", ErrorCode.CONFIG_ERROR)
        del self._solvers[name]

    def get(self, name: str) -> Solver:
        try:
            return self._solvers[name]
        except KeyError:
            raise RippleCutError(f"solver '{name}' is not registered", ErrorCode.CONFIG_ERROR) from None

    def maybe_get(self, name: str) -> Optional[Solver]:
        return self._solvers.get(name)

    def list_available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._solvers))

    def __contains__(self, name: object) -> bool:
        return name in self._solvers

    def __len__(self) -> int:
        return len(self._solvers)

    def with_override(self, solver: Solver) -> "SolverRegistry":
        """A copy in which ``solver`` replaces the entry with the same name (used by fault injection)."""
        copy = SolverRegistry(self._solvers.values())
        copy.register(solver, replace=True)
        return copy

    def describe(self) -> List[dict]:
        return [{"name": n, "kind": s.capabilities.kind.value, "max_actions": s.capabilities.max_actions,
                 "description": s.capabilities.description} for n, s in sorted(self._solvers.items())]
