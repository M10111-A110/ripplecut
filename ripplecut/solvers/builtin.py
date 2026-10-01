"""Built-in solver registration.

This is the ONLY module that names concrete solver classes. Everything else
(pipeline, orchestrator, validator, UI, CLI) receives a ``SolverRegistry`` and
a ``SolverPolicy`` and never refers to a specific algorithm (master spec §4,
§122). Adding a solver = implement ``Solver`` + register it here (or in any
caller-built registry) + optionally list it in config/solver_policy.json.
"""
from __future__ import annotations

from .branch_and_bound import BranchAndBoundSolver
from .exhaustive import ExhaustiveSolver
from .registry import SolverRegistry


def build_default_registry() -> SolverRegistry:
    return SolverRegistry([
        BranchAndBoundSolver(strategy="best_first", name="branch_and_bound"),
        BranchAndBoundSolver(strategy="depth_first", name="branch_and_bound_dfs"),
        ExhaustiveSolver(),
    ])
