"""Exact Branch & Bound for the lexicographic containment objective (master spec §36-§41).

Search space: the same 2^m subsets B of A as the exhaustive oracle, organised
as a set-enumeration tree. Actions are ordered by (cost, id); node B with last
index j has children B + {a_k} for k > j, so every subset occurs exactly once.

Order being minimised (identical to the oracle, from engine.objective):
    feasible only, then key(B) = (K(B), N(B), R(B), sorted ids)

Pruning rules — every rule discards only subtrees whose members are provably
illegal, provably infeasible, or provably strictly worse than the incumbent:

  (S0) Precondition filter. An action whose preconditions fail on x(0) makes
       every plan containing it illegal; such actions are never branched on.
  (S1) Upward-closed illegality. If B is illegal for an upward-closed reason
       (unknown/duplicate action, INVALID_COMBINATION, ambiguous higher-order
       joint effect), every descendant is illegal.
  (S2) Cost / count bound. Every descendant D of B satisfies
       K(D) >= K(B) + min remaining cost   (costs are >= 0, D adds >= 1 action)
       N(D) >= N(B) + 1.
       Componentwise lower bounds imply a lexicographic lower bound, so if
       (LB_K, LB_N) >lex (K*, N*) the whole subtree is strictly worse.
  (S3) Critical-feasibility upper bound UB_C. The optimistic state sets UP every
       service that any unit of any legal descendant could set UP and applies
       no DOWN effect; hence T_D(x0) <= x_opt for every legal descendant D.
       Phi is monotone, so Cascade(T_D(x0)) <= Cascade(x_opt) and
       C(D) <= UB_C. If UB_C < C_min, no descendant is feasible.
  (S4) Residual lower bound LB_R from the same optimistic state
       (R(D) >= R(Cascade(x_opt))). If (LB_K, LB_N, LB_R) >lex (K*, N*, R*),
       the subtree is strictly worse. Equality never prunes (ids may still win).

Not used for pruning: the per-action count bound
    LB_N = N + ceil(remaining critical failures / max recoveries per action)
is UNSAFE under AND rules because recoveries are super-additive (two actions
can jointly recover a service neither recovers alone). See docs/MATHEMATICS.md
and tests/test_branch_and_bound.py::test_naive_count_bound_is_unsafe.

Two traversal strategies are provided (both exact, both verified against the
oracle): "best_first" pops nodes in (K, N, ids) order and stops as soon as the
next node is strictly worse than the incumbent; "depth_first" is the classic DFS.
"""
from __future__ import annotations

import heapq
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from ..engine.objective import PlanEvaluation
from ..errors import ResourceLimitExceeded, SolverTimeout
from ..model.schema import ContainmentProblem
from .base import (OptimalityStatus, Solver, SolverCapabilities, SolverContext, SolverKind, SolverResult,
                   SolverStatus, result_from_evaluation)


class BranchAndBoundSolver(Solver):
    capabilities = SolverCapabilities(kind=SolverKind.EXACT, max_actions=None,
                                      description="Exact lexicographic Branch & Bound with safe bounds (S0-S4).")

    def __init__(self, strategy: str = "best_first", name: str = "branch_and_bound",
                 use_feasibility_bound: bool = True, use_residual_bound: bool = True) -> None:
        if strategy not in ("best_first", "depth_first"):
            raise ValueError("strategy must be 'best_first' or 'depth_first'")
        self.strategy = strategy
        self.name = name
        self.use_feasibility_bound = use_feasibility_bound
        self.use_residual_bound = use_residual_bound

    # ------------------------------------------------------------------------
    def solve(self, problem: ContainmentProblem, context: SolverContext) -> SolverResult:
        universe = problem.actions
        applicable = set(context.applicable_actions())                               # (S0)
        order = sorted(applicable, key=lambda a: (universe.get(a).cost, a))
        costs = [universe.get(a).cost for a in order]
        c_min = problem.system.c_min
        stats: Dict[str, int] = {"nodes_evaluated": 0, "pruned_illegal_S1": 0, "pruned_cost_count_S2": 0,
                                 "pruned_infeasible_S3": 0, "pruned_residual_S4": 0,
                                 "stopped_by_best_first_order": 0, "max_frontier": 0}
        best: Optional[PlanEvaluation] = None
        status, complete = SolverStatus.SUCCESS, True

        def expand_ok(idx_plan: Tuple[int, ...], last: int, k_plan: Fraction) -> bool:
            """Whether B's strict descendants may contain an improvement (S2-S4)."""
            if last + 1 >= len(order):
                return False
            lb_k = k_plan + costs[last + 1]          # sorted ascending: min remaining cost
            lb_n = len(idx_plan) + 1
            if best is not None and (lb_k, lb_n) > (best.objective.cost, best.objective.count):
                stats["pruned_cost_count_S2"] += 1
                return False
            if self.use_feasibility_bound or self.use_residual_bound:
                bound = context.optimistic_bound([order[i] for i in idx_plan],
                                                 [order[i] for i in range(last + 1, len(order))])
                if self.use_feasibility_bound and bound.critical_upper < c_min:
                    stats["pruned_infeasible_S3"] += 1
                    return False
                if (self.use_residual_bound and best is not None and
                        (lb_k, lb_n, bound.residual_lower) >
                        (best.objective.cost, best.objective.count, best.objective.residual)):
                    stats["pruned_residual_S4"] += 1
                    return False
            return True

        def visit(idx_plan: Tuple[int, ...]) -> bool:
            """Evaluate node B; return False if its subtree is illegal (S1)."""
            nonlocal best
            ev = context.evaluate([order[i] for i in idx_plan])
            stats["nodes_evaluated"] += 1
            if not ev.legal:
                if ev.legality.upward_closed:
                    stats["pruned_illegal_S1"] += 1
                    return False
                return True
            if ev.feasible and (best is None or ev.key < best.key):
                best = ev
            return True

        try:
            if self.strategy == "depth_first":
                stack: List[Tuple[Tuple[int, ...], int, Fraction]] = [((), -1, Fraction(0))]
                while stack:
                    stats["max_frontier"] = max(stats["max_frontier"], len(stack))
                    idx_plan, last, k_plan = stack.pop()
                    if visit(idx_plan) and expand_ok(idx_plan, last, k_plan):
                        for k in range(len(order) - 1, last, -1):
                            stack.append((idx_plan + (k,), k, k_plan + costs[k]))
            else:
                heap: List[Tuple[Fraction, int, Tuple[str, ...], Tuple[int, ...], int]] = [
                    (Fraction(0), 0, (), (), -1)]
                while heap:
                    stats["max_frontier"] = max(stats["max_frontier"], len(heap))
                    k_plan, n_plan, _ids, idx_plan, last = heapq.heappop(heap)
                    if best is not None and (k_plan, n_plan) > (best.objective.cost, best.objective.count):
                        # Every remaining node (and its subtree) has key >= (k_plan, n_plan) >lex (K*, N*).
                        stats["stopped_by_best_first_order"] = 1 + len(heap)
                        break
                    context.check_limits()
                    if visit(idx_plan) and expand_ok(idx_plan, last, k_plan):
                        for k in range(last + 1, len(order)):
                            child = idx_plan + (k,)
                            heapq.heappush(heap, (k_plan + costs[k], n_plan + 1,
                                                  tuple(sorted(order[i] for i in child)), child, k))
        except SolverTimeout:
            status, complete = SolverStatus.TIMEOUT, False
        except ResourceLimitExceeded:
            status, complete = SolverStatus.RESOURCE_LIMIT, False

        meta = {**stats, **context.stats(), "strategy": self.strategy, "m": problem.m,
                "candidates_total": 2 ** problem.m,
                "precondition_excluded_actions": sorted(set(universe.ids) - applicable),
                "branching_order": order, "search_complete": complete}
        if best is None:
            if complete:
                return SolverResult(self.name, SolverStatus.NO_FEASIBLE_PLAN, search_complete=True,
                                    nodes_explored=stats["nodes_evaluated"],
                                    optimality_status=OptimalityStatus.PROVEN_OPTIMAL, metadata=meta)
            return SolverResult(self.name, status, search_complete=False, nodes_explored=stats["nodes_evaluated"],
                                metadata=meta, error="stopped before any feasible plan was found")
        return result_from_evaluation(
            self.name, SolverStatus.SUCCESS if complete else status, best, search_complete=complete,
            optimality=OptimalityStatus.PROVEN_OPTIMAL if complete else OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL,
            nodes_explored=stats["nodes_evaluated"], metadata=meta, system_ids=problem.system.service_ids)
