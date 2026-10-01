"""Exhaustive enumeration — the reference / oracle solver (master spec §42-§43, PDF §7.3).

Evaluates every one of the 2^m subsets B of the action universe A through the
canonical evaluator and keeps the lexicographic minimum of
(K, N, R, sorted ids) over legal feasible plans. It deliberately contains no
pruning at all: correctness and clarity over speed.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional

from ..engine.objective import PlanEvaluation
from ..errors import ResourceLimitExceeded, SolverTimeout
from ..model.schema import ContainmentProblem, cost_to_json
from .base import (OptimalityStatus, Solver, SolverCapabilities, SolverContext, SolverKind, SolverResult,
                   SolverStatus, result_from_evaluation)

TOP_K = 5


class ExhaustiveSolver(Solver):
    name = "exhaustive"
    capabilities = SolverCapabilities(kind=SolverKind.EXACT, max_actions=16,
                                      description="Oracle: evaluates all 2^m candidate subsets; no pruning.")

    def solve(self, problem: ContainmentProblem, context: SolverContext) -> SolverResult:
        ids = sorted(problem.actions.ids)
        m = len(ids)
        best: Optional[PlanEvaluation] = None
        top: List[PlanEvaluation] = []
        counts = {"candidates_total": 2 ** m, "legal": 0, "feasible": 0, "illegal": 0}
        status, complete = SolverStatus.SUCCESS, True
        try:
            for r in range(m + 1):
                for combo in combinations(ids, r):
                    ev = context.evaluate(combo)
                    if not ev.legal:
                        counts["illegal"] += 1
                        continue
                    counts["legal"] += 1
                    if not ev.feasible:
                        continue
                    counts["feasible"] += 1
                    top.append(ev)
                    top.sort(key=lambda e: e.key)
                    del top[TOP_K:]
                    if best is None or ev.key < best.key:
                        best = ev
        except SolverTimeout:
            status, complete = SolverStatus.TIMEOUT, False
        except ResourceLimitExceeded:
            status, complete = SolverStatus.RESOURCE_LIMIT, False

        meta = {**counts, **context.stats(), "search_complete": complete,
                "top_feasible": [{"plan": list(e.plan), "K": cost_to_json(e.objective.cost), "N": e.objective.count,
                                  "R": e.objective.residual} for e in top]}
        if best is None:
            if complete:
                return SolverResult(self.name, SolverStatus.NO_FEASIBLE_PLAN, search_complete=True,
                                    nodes_explored=context.candidates_examined,
                                    optimality_status=OptimalityStatus.PROVEN_OPTIMAL, metadata=meta)
            return SolverResult(self.name, status, search_complete=False,
                                nodes_explored=context.candidates_examined, metadata=meta,
                                error="stopped before any feasible plan was found")
        final_status = SolverStatus.SUCCESS if complete else status
        claim = OptimalityStatus.PROVEN_OPTIMAL if complete else OptimalityStatus.VALID_NOT_PROVEN_OPTIMAL
        return result_from_evaluation(self.name, final_status, best, search_complete=complete, optimality=claim,
                                      nodes_explored=context.candidates_examined, metadata=meta,
                                      system_ids=problem.system.service_ids)
