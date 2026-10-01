"""Deterministic explanation engine (master spec §70; PDF §14.3).

Every sentence is derived from computed data: the uncontrolled cascade trace,
the dependency rules, the orchestration record, the validator's checks and a
small set of comparison plans evaluated through RippleCut's own engine.

The comparison plans (the empty plan, every plan with at most two actions and
every one-action neighbour of the selected plan, capped) exist only to explain
the lexicographic decision in concrete terms. They are NOT the search: the
optimality statement always comes from the orchestration result, which in turn
comes from an exact solver that completed its search (or is reported as not
proven).

No LLM is involved here. An optional LLM may rephrase the resulting facts; see
``ripplecut.explain.llm_explainer`` for the guard applied to such text.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..engine.dependency import describe_rule
from ..engine.objective import PlanEvaluation, evaluate_plan
from ..engine.simulator import CascadeResult
from ..model.schema import ContainmentProblem, cost_to_json

MAX_COMPARISONS = 300
CRITERIA = ("total cost K", "number of interventions N", "residual failed services R", "sorted action ids")


def _plan_label(plan: Sequence[str]) -> str:
    return "{" + ", ".join(plan) + "}" if plan else "{} (do nothing)"


def deciding_criterion(chosen: PlanEvaluation, other: PlanEvaluation) -> str:
    """Which component of (K, N, R, ids) separates ``other`` from the chosen plan."""
    a, b = chosen.key, other.key
    for i, name in enumerate(CRITERIA):
        if a[i] != b[i]:
            if i == 0:
                return f"higher {name} ({cost_to_json(b[0])} vs {cost_to_json(a[0])})"
            if i < 3:
                return f"same K, but higher {name} ({b[i]} vs {a[i]})" if i == 1 else \
                    f"same K and N, but higher {name} ({b[i]} vs {a[i]})"
            return "identical (K, N, R); loses only on the deterministic sorted-id tie-break"
    return "identical"


def comparison_plans(problem: ContainmentProblem, selected: Sequence[str]) -> List[Tuple[str, ...]]:
    ids = sorted(problem.actions.ids)
    plans: List[Tuple[str, ...]] = [()]
    for r in (1, 2):
        plans += [tuple(c) for c in combinations(ids, r)]
    sel = set(selected)
    for a in ids:
        neighbour = tuple(sorted(sel ^ {a}))
        if neighbour not in plans:
            plans.append(neighbour)
    seen, out = set(), []
    for p in plans:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:MAX_COMPARISONS]


def cascade_chain(problem: ContainmentProblem, cascade: CascadeResult) -> List[Dict[str, Any]]:
    system = problem.system
    chain = []
    prev = cascade.history[0].state
    for rnd in cascade.history[1:]:
        items = []
        for s in rnd.newly_failed:
            rule = system.rules.get(s)
            down_inputs = [u for u in (rule.hard_upstream if rule else ()) if prev[system.index(u)] == 0]
            items.append({"service": s, "critical": s in system.critical, "because_down": down_inputs,
                          "rule": describe_rule(rule) if rule else "", "rule_label": "RIPPLECUT-MODELED RULE"})
        chain.append({"round": rnd.round, "newly_failed": items})
        prev = rnd.state
    return chain


def explain(problem: ContainmentProblem, uncontrolled: CascadeResult, orchestration: Any,
            contained: Optional[PlanEvaluation], infeasibility: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    system, universe = problem.system, problem.actions
    initial_down = list(system.down_services(uncontrolled.initial_state))
    final_down = list(system.down_services(uncontrolled.final_state))
    critical_lost = [s for s in final_down if s in system.critical]
    chain = cascade_chain(problem, uncontrolled)
    out: Dict[str, Any] = {
        "initial_failures": initial_down,
        "cascade": chain,
        "uncontrolled_final_down": final_down,
        "uncontrolled_critical_down": critical_lost,
        "status": orchestration.status,
        "solver": {"selected": orchestration.selected_solver, "fallback_count": orchestration.fallback_count,
                   "attempts": [{"solver": a.solver_name, "status": a.result.status.value, "outcome": a.outcome,
                                 "error": a.result.error} for a in orchestration.attempts]},
        "optimality": {"status": orchestration.optimality_status.value, "basis": orchestration.optimality_basis},
        "verification": {"validator": orchestration.validation.reason if orchestration.validation else None,
                         "checks_passed": sum(c.passed for c in orchestration.validation.checks)
                         if orchestration.validation else 0,
                         "checks_total": len(orchestration.validation.checks) if orchestration.validation else 0,
                         "cross_check": orchestration.verification.get("status")},
        "approval": "Recommendation only. A human must approve before any real-world action; RippleCut executes nothing.",
    }
    lines: List[str] = []
    lines.append("Initial failure(s): " + (", ".join(initial_down) if initial_down else "none"))
    if chain:
        lines.append("Cascade under the explicit model (RIPPLECUT-MODELED rules):")
        for rnd in chain:
            for it in rnd["newly_failed"]:
                lines.append(f"  round {rnd['round']}: {it['service']} became unavailable because hard input(s) "
                             f"{', '.join(it['because_down'])} were DOWN  [{it['rule']}]")
    else:
        lines.append("Cascade: no further service fails under the modeled rules (already a fixed point).")
    if critical_lost:
        lines.append("Without intervention, critical services DOWN: " + ", ".join(critical_lost))
    else:
        lines.append("Without intervention, every designated critical service stays UP.")

    if orchestration.status == "SUCCESS" and contained is not None and contained.objective is not None:
        obj = contained.objective
        plan = list(orchestration.plan or ())
        out["selected_plan"] = [{"id": a, "name": universe.get(a).name, "cost": cost_to_json(universe.get(a).cost),
                                 "label": universe.get(a).metadata.get("label", "RIPPLECUT-MODELED")} for a in plan]
        out["objective"] = obj.to_dict()
        comparisons = []
        for p in comparison_plans(problem, plan):
            if p == tuple(sorted(plan)):
                continue
            ev = evaluate_plan(problem, p)
            if not ev.legal:
                comparisons.append({"plan": list(p), "legal": False, "verdict": f"illegal: {ev.legality.code}",
                                    "detail": ev.legality.detail})
            elif not ev.feasible:
                comparisons.append({"plan": list(p), "legal": True, "feasible": False,
                                    "verdict": "infeasible: critical DOWN " + ", ".join(ev.objective.critical_down)
                                    if ev.objective else "cascade did not reach a fixed point"})
            else:
                comparisons.append({"plan": list(p), "legal": True, "feasible": True,
                                    "K": cost_to_json(ev.objective.cost), "N": ev.objective.count,
                                    "R": ev.objective.residual, "key": ev.key,
                                    "verdict": deciding_criterion(contained, ev)})
        feasible_alts = sorted([c for c in comparisons if c.get("feasible")], key=lambda c: c["key"])
        for c in feasible_alts:
            c.pop("key", None)
        out["runner_ups"] = feasible_alts[:3]
        out["comparisons_evaluated"] = len(comparisons) + 1
        empty = next((c for c in comparisons if c["plan"] == []), None)
        lines.append("Selected containment: " + (_plan_label(plan) if plan else "do nothing (empty plan)"))
        for a in out["selected_plan"]:
            lines.append(f"  - {a['id']}: {a['name']} (cost {a['cost']}; {a['label']})")
        lines.append(f"Outcome after re-running the cascade: C = {obj.critical_up}/{obj.critical_required} critical "
                     f"UP, K = {cost_to_json(obj.cost)}, N = {obj.count}, R = {obj.residual}"
                     + (f" (still DOWN: {', '.join(obj.residual_services)})" if obj.residual_services else ""))
        lines.append("Why this plan: the policy is lexicographic (PDF §8.3) - keep every critical service UP, then "
                     "minimise total cost K, then number of interventions N, then residual failures R.")
        if plan and empty is not None:
            lines.append(f"  Doing nothing is {empty['verdict']}.")
        if feasible_alts:
            for c in feasible_alts[:3]:
                lines.append(f"  Runner-up {_plan_label(c['plan'])} = (K={c['K']}, N={c['N']}, R={c['R']}): "
                             f"{c['verdict']}.")
        illegal = [c for c in comparisons if not c["legal"] and len(c["plan"]) == 2]
        if illegal:
            lines.append(f"  {len(illegal)} two-action combination(s) are rejected before simulation "
                         f"(conflicts or unmet preconditions), e.g. {_plan_label(illegal[0]['plan'])}: "
                         f"{illegal[0]['verdict']}.")
        if orchestration.optimality_status.value == "PROVEN_OPTIMAL":
            lines.append("  Optimality: PROVEN_OPTIMAL within the finite modeled action set - "
                         + orchestration.optimality_basis + ".")
        else:
            lines.append(f"  Optimality: {orchestration.optimality_status.value} - {orchestration.optimality_basis}. "
                         "No minimality claim is made.")
    elif orchestration.status == "NO_FEASIBLE_PLAN":
        info = infeasibility or {}
        crit = info.get("optimistic_critical_down", [])
        x0 = uncontrolled.initial_state
        unreachable = [s for s in crit if x0[system.index(s)] == 0
                       and not any(s in a.effect.set_up for a in universe.interventions)]
        out["infeasibility"] = {**info, "critical_with_no_restoring_action": unreachable}
        lines.append("NO_FEASIBLE_PLAN: the configured intervention set cannot preserve all designated critical "
                     "services under the current RippleCut model.")
        lines.append("  Basis: " + orchestration.optimality_basis + ".")
        if crit:
            lines.append("  Even applying every available UP effect at once leaves critical services DOWN: "
                         + ", ".join(crit) + ".")
        if unreachable:
            lines.append("  Blocking initial failure(s) that no configured action sets UP: " + ", ".join(unreachable)
                         + "; the other critical losses follow from them through the cascade.")
        lines.append("  The critical-service constraint is never relaxed; RippleCut recommends no plan.")
    else:
        lines.append("No valid solver result: every configured solver failed or was rejected by the independent "
                     "validator. RippleCut does not fabricate a plan.")
    lines.append("Solver: " + (orchestration.selected_solver or "none accepted")
                 + (f" (after {orchestration.fallback_count} fallback(s))" if orchestration.fallback_count else ""))
    for a in orchestration.attempts:
        lines.append(f"  attempt {a.solver_name}: {a.result.status.value} -> {a.outcome}"
                     + (f" ({a.result.error})" if a.result.error and a.outcome != "ACCEPTED" else ""))
    if orchestration.validation is not None:
        v = out["verification"]
        lines.append(f"Verification: independent validator - {v['validator']} "
                     f"({v['checks_passed']}/{v['checks_total']} checks passed); cross-check: {v['cross_check']}.")
    lines.append("Approval: " + out["approval"])
    out["text"] = "\n".join(lines)
    return out
