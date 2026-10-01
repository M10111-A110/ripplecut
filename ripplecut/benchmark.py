"""Exhaustive vs Branch & Bound benchmark (master spec §94-§95). Measured values only.

Two parts:
  1. every canonical scenario (m = 8, 2^m = 256 candidates);
  2. seeded random instances of growing m (the oracle is run up to m = 16;
     above that only B&B is run and no equivalence claim is made).

Every solver call goes through the same ExecutionGuard used in production and
every result is re-validated by the SafetyValidator. Runtimes are medians of
``repeats`` runs measured by the guard (wall clock, includes thread start).
The benchmark names concrete solvers on purpose: comparing them is its job.
"""
from __future__ import annotations

import datetime as _dt
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .generators import random_problem
from .model.schema import ContainmentProblem, ResourceLimits, cost_to_json
from .solvers.base import Solver, SolverKind, SolverResult, SolverStatus
from .solvers.branch_and_bound import BranchAndBoundSolver
from .solvers.exhaustive import ExhaustiveSolver
from .solvers.guard import ExecutionGuard
from .validation.validator import SafetyValidator

BENCH_LIMITS = ResourceLimits(timeout_seconds=120, max_evaluations=50_000_000, grace_seconds=1.0)

# Two seeded instance families. "mixed": wide cost range, 1-3 initial failures.
# "tight": near-uniform costs (weak cost bound), 50% critical services, 3-4 initial failures, and every
#          initially failed service is restorable by at least one action (planted), so more instances are feasible.
FAMILIES: Dict[str, Dict[str, Any]] = {
    "mixed": {"n_services": 14},
    "tight": {"n_services": 14, "crit_frac": 0.5, "cost_choices": (1, 1, 2), "n_failures": (3, 4),
              "plant_restorers": True},
}


def _measure(solver: Solver, problem: ContainmentProblem, repeats: int) -> Tuple[SolverResult, float]:
    guard = ExecutionGuard()
    runs, last = [], None
    for _ in range(repeats):
        last = guard.run(solver, problem, BENCH_LIMITS)
        runs.append(last.runtime_seconds)
    return last, statistics.median(runs)


def _key(problem: ContainmentProblem, res: SolverResult, validator: SafetyValidator) -> Tuple[Any, float, bool]:
    t0 = time.perf_counter()
    val = validator.validate(problem, res, SolverKind.EXACT)
    dt = time.perf_counter() - t0
    if res.status is SolverStatus.NO_FEASIBLE_PLAN:
        return ("NO_FEASIBLE_PLAN",), dt, val.valid
    if res.status is not SolverStatus.SUCCESS or val.recomputed_objective is None:
        return (res.status.value,), dt, val.valid
    o = val.recomputed_objective
    return ((cost_to_json(o.cost), o.count, o.residual), tuple(res.selected_plan)), dt, val.valid


def compare(problem: ContainmentProblem, repeats: int, run_oracle: bool = True) -> Dict[str, Any]:
    validator = SafetyValidator()
    m = problem.m
    row: Dict[str, Any] = {"m": m, "candidates": 2 ** m}
    bb, bb_t = _measure(BranchAndBoundSolver("best_first"), problem, repeats)
    dfs, dfs_t = _measure(BranchAndBoundSolver("depth_first", name="branch_and_bound_dfs"), problem, repeats)
    bb_key, val_t, bb_valid = _key(problem, bb, validator)
    dfs_key, _, dfs_valid = _key(problem, dfs, validator)
    md = dict(bb.metadata)
    row.update({
        "bb_status": bb.status.value, "bb_runtime_ms": bb_t * 1e3, "bb_nodes": bb.nodes_explored,
        "bb_plan_simulations": md.get("plan_simulations"), "bb_bound_simulations": md.get("bound_simulations"),
        "bb_pruning_ratio": 1 - (bb.nodes_explored or 0) / 2 ** m,
        "bb_prunes": {k: md.get(k) for k in ("pruned_illegal_S1", "pruned_cost_count_S2", "pruned_infeasible_S3",
                                             "pruned_residual_S4", "stopped_by_best_first_order")},
        "dfs_runtime_ms": dfs_t * 1e3, "dfs_nodes": dfs.nodes_explored,
        "objective": bb_key[0], "plan": list(bb_key[1]) if len(bb_key) > 1 else None,
        "validation_ms": val_t * 1e3, "validator_accepts_bb": bb_valid, "validator_accepts_dfs": dfs_valid,
        "dfs_equals_bb": dfs_key == bb_key,
    })
    if run_oracle:
        ex, ex_t = _measure(ExhaustiveSolver(), problem, repeats)
        ex_key, _, ex_valid = _key(problem, ex, validator)
        row.update({"ex_status": ex.status.value, "ex_runtime_ms": ex_t * 1e3,
                    "ex_simulations": dict(ex.metadata).get("plan_simulations"),
                    "validator_accepts_ex": ex_valid, "equivalent": ex_key == bb_key == dfs_key,
                    "speedup": (ex_t / bb_t) if bb_t > 0 else None})
    else:
        row.update({"ex_status": "NOT_RUN", "equivalent": None})
    return row


def run_benchmark(app: Any, repeats: int = 5, seeds: int = 20, quick: bool = False) -> Dict[str, Any]:
    started = time.perf_counter()
    from .pipeline import REPO_ROOT  # local import keeps the benchmark importable on its own
    from .evidence.sources import ReplayFixtureSource
    scenario_rows = []
    for s in app.scenarios:
        if "fixture" in s:
            b = ReplayFixtureSource(REPO_ROOT / s["fixture"]).load()
            x0 = app.estimator.estimate(app.bundle.system, b.observations, b.label).formal_state(app.bundle.system)
        else:
            x0 = app.bundle.system.state_with_failures(s["incident"]["failed_services"] +
                                                        s["incident"].get("degraded_services", []))
        problem = ContainmentProblem(problem_id=s["id"], system=app.bundle.system, actions=app.bundle.actions,
                                     initial_state=x0, objective=app.bundle.objective, resource_limits=BENCH_LIMITS)
        scenario_rows.append({"scenario": s["id"], **compare(problem, repeats)})

    sizes = [8, 10, 12] if quick else [8, 10, 12, 14, 16]
    random_rows: List[Dict[str, Any]] = []
    n_seeds = min(seeds, 5) if quick else seeds
    for family, kw in FAMILIES.items():
        for m in sizes:
            per_size = n_seeds if m <= 14 else max(3, n_seeds // 4)
            for i in range(per_size):
                seed = 1000 * m + i
                problem, _ = random_problem(seed, m_actions=m, limits=BENCH_LIMITS, **kw)
                random_rows.append({"family": family, "seed": seed, **compare(problem, 1)})
    beyond = []
    for m in ([20] if quick else [20, 24]):
        for seed in range(3):
            for family, kw in FAMILIES.items():
                problem, _ = random_problem(1000 * m + seed, m_actions=m, limits=BENCH_LIMITS, **kw)
                beyond.append({"family": family, "seed": 1000 * m + seed, **compare(problem, 1, run_oracle=False)})

    def agg(rows: List[Dict[str, Any]], family: str, m: int) -> Dict[str, Any]:
        rs = [r for r in rows if r["m"] == m and r["family"] == family]
        return {"family": family, "m": m, "instances": len(rs), "candidates": 2 ** m,
                "feasible": sum(1 for r in rs if r["objective"] != "NO_FEASIBLE_PLAN"),
                "equivalent": sum(1 for r in rs if r.get("equivalent")),
                "ex_median_ms": statistics.median(r["ex_runtime_ms"] for r in rs) if "ex_runtime_ms" in rs[0] else None,
                "bb_median_ms": statistics.median(r["bb_runtime_ms"] for r in rs),
                "bb_median_nodes": statistics.median(r["bb_nodes"] for r in rs),
                "bb_max_nodes": max(r["bb_nodes"] for r in rs),
                "bb_median_pruning": statistics.median(r["bb_pruning_ratio"] for r in rs)}

    all_rows = scenario_rows + random_rows
    summary = {
        "scenario_instances": len(scenario_rows),
        "random_instances_with_oracle": len(random_rows),
        "all_equivalent": all(r["equivalent"] for r in all_rows),
        "non_equivalent": [r.get("scenario", f"{r.get('family')}:{r.get('seed')}") for r in all_rows
                           if not r["equivalent"]],
        "validator_rejections": sum(1 for r in all_rows + beyond if not (r["validator_accepts_bb"] and
                                                                         r["validator_accepts_dfs"] and
                                                                         r.get("validator_accepts_ex", True))),
        "benchmark_seconds": round(time.perf_counter() - started, 2),
    }
    return {
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "platform": platform.platform(), "machine": platform.machine(),
                        "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        "ripplecut": __version__, "repeats": repeats, "quick": quick},
        "scenarios": scenario_rows, "random": random_rows,
        "random_by_size": [agg(random_rows, f, m) for f in FAMILIES for m in sizes],
        "beyond_oracle": beyond, "summary": summary,
    }


def _fmt(v: Any, nd: int = 2) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, tuple):
        return "(" + ", ".join(str(x) for x in v) + ")"
    return str(v)


def write_markdown(data: Dict[str, Any], path: Path) -> None:
    env, s = data["environment"], data["summary"]
    L: List[str] = []
    L.append("# RippleCut Benchmark — Exhaustive vs Branch & Bound\n")
    L.append("Generated by `python -m ripplecut benchmark --out docs/BENCHMARK.md`. **Every number below was "
             "measured by that run; none is estimated.** Re-running on another machine will give different "
             "runtimes; node counts and objectives are deterministic.\n")
    L.append(f"Environment: Python {env['python']} ({env['implementation']}), {env['platform']}, "
             f"{env['machine']}; {env['date_utc']}; RippleCut {env['ripplecut']}; runtime = median of "
             f"{env['repeats']} guarded runs for scenarios, 1 run for random instances.\n")
    L.append("Definitions: *nodes* = candidate plans B evaluated by B&B (legality check, and cascade simulation "
             "if legal); *bound sims* = optimistic-state cascade simulations used by bounds S3/S4; "
             "*pruning ratio* = 1 − nodes / 2^m; *equivalent* = identical validated objective tuple (K, N, R) "
             "**and** identical plan for Exhaustive, B&B best-first and B&B depth-first (or all three report "
             "NO_FEASIBLE_PLAN).\n")
    L.append(f"**Summary:** {s['scenario_instances']} canonical scenarios + {s['random_instances_with_oracle']} "
             f"seeded random instances compared against the oracle; all equivalent: **{s['all_equivalent']}**"
             f"{' (non-equivalent: ' + str(s['non_equivalent']) + ')' if s['non_equivalent'] else ''}; "
             f"validator rejections of solver output: {s['validator_rejections']}; total benchmark time "
             f"{s['benchmark_seconds']} s.\n")
    L.append("## 1. Canonical Online Boutique scenarios (m = 8, 2^m = 256)\n")
    L.append("| scenario | objective (K,N,R) | plan | Exhaustive ms | B&B ms | B&B nodes | bound sims | pruning | "
             "DFS nodes | validation ms | equivalent |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["scenarios"]:
        L.append(f"| {r['scenario']} | {_fmt(r['objective'])} | {', '.join(r['plan']) if r['plan'] else ('—' if r['plan'] is None else '∅')} | "
                 f"{_fmt(r['ex_runtime_ms'])} | {_fmt(r['bb_runtime_ms'])} | {r['bb_nodes']} | "
                 f"{r['bb_bound_simulations']} | {_fmt(r['bb_pruning_ratio'], 3)} | {r['dfs_nodes']} | "
                 f"{_fmt(r['validation_ms'], 3)} | {'YES' if r['equivalent'] else 'NO'} |")
    L.append("\nPrune counters (best-first B&B) per scenario:\n")
    L.append("| scenario | S1 illegal | S2 cost/count | S3 infeasible | S4 residual | stopped by best-first order |")
    L.append("|---|---|---|---|---|---|")
    for r in data["scenarios"]:
        p = r["bb_prunes"]
        L.append(f"| {r['scenario']} | {p['pruned_illegal_S1']} | {p['pruned_cost_count_S2']} | "
                 f"{p['pruned_infeasible_S3']} | {p['pruned_residual_S4']} | {p['stopped_by_best_first_order']} |")
    L.append("\n## 2. Seeded random instances (14 services, cycles, OR/THRESHOLD/GROUP rules, conflicts, joint "
             "effects)\n")
    L.append("Seeds are `1000*m + i`; instances are produced by `ripplecut.generators.random_problem`. Families: "
             "*mixed* = costs drawn from {0,1,1,2,2,3,4,5,7}, 40% critical, 1-3 initial failures; *tight* = costs "
             "from {1,1,2} (weak cost bound), 50% critical, 3-4 initial failures, every initially failed service "
             "restorable by at least one (planted) action.\n")
    L.append("| family | m | 2^m | instances | feasible | equivalent | Exhaustive median ms | B&B median ms | "
             "B&B median nodes | B&B max nodes | median pruning |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in data["random_by_size"]:
        L.append(f"| {a['family']} | {a['m']} | {a['candidates']} | {a['instances']} | {a['feasible']} | {a['equivalent']} | "
                 f"{_fmt(a['ex_median_ms'])} | {_fmt(a['bb_median_ms'])} | {a['bb_median_nodes']} | "
                 f"{a['bb_max_nodes']} | {_fmt(a['bb_median_pruning'], 3)} |")
    L.append("\n## 3. Beyond the oracle range (B&B only — no equivalence claim)\n")
    L.append("| family | seed | m | 2^m | status | objective | B&B ms | nodes | pruning | DFS agrees with best-first |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in data["beyond_oracle"]:
        L.append(f"| {r['family']} | {r['seed']} | {r['m']} | {r['candidates']} | {r['bb_status']} | {_fmt(r['objective'])} | "
                 f"{_fmt(r['bb_runtime_ms'])} | {r['bb_nodes']} | {_fmt(r['bb_pruning_ratio'], 4)} | "
                 f"{'YES' if r['dfs_equals_bb'] else 'NO'} |")
    L.append("\n## Reading these numbers honestly\n")
    L.append("* Equivalence is verified only where the oracle ran. Above m = 16 the two B&B strategies agreeing is "
             "a consistency check, not a proof; the proof of B&B exactness is the safety argument in "
             "docs/MATHEMATICS.md, and the empirical check is the table above plus the randomized tests.\n"
             "* A node count of 1 means the root decided the instance: either the S3 bound at the root is the "
             "validator's infeasibility certificate (instance infeasible), or the empty plan is feasible and "
             "nothing can beat K = 0, N = 0. Such instances inflate the median pruning ratio; the max-nodes column "
             "shows the harder ones.\n"
             "* These instances are easy for B&B because optimal plans are small. The worst case is still 2^m "
             "nodes; no sub-exponential guarantee is claimed.\n"
             "* Pruning is instance-dependent. The canonical scenarios have many precondition-excluded actions, so "
             "the S0 filter alone removes most of the tree.\n"
             "* B&B node counts include illegal nodes it had to evaluate to discover illegality; bound simulations "
             "are extra cascade runs Exhaustive does not need, so fewer nodes does not always mean less work at "
             "tiny m.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
