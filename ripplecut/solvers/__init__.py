"""Solver framework: interface, registry, policy, execution guard, orchestrator and solvers.

Kept import-free on purpose (avoids import cycles); import submodules directly,
e.g. ``from ripplecut.solvers.registry import SolverRegistry``.
"""
