"""Seeded random instances for property tests and benchmarks (master spec §44, §46, §63).

Every instance is built as a raw configuration dictionary and passed through the
normal loader + model validation, so generated models obey exactly the same
rules as the canonical model (no shortcut). Randomness is always explicit:
``random.Random(seed)``; the seed is recorded in the problem metadata.

Generated models deliberately include what the canonical model lacks: OR,
THRESHOLD and GROUP rules, dependency cycles, set_down effects, preconditions,
INVALID_COMBINATION conflicts and DEFINED_JOINT_EFFECT pairs.
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import Any, Dict, List, Tuple

from .model.loader import ModelBundle, parse_model
from .model.schema import ContainmentProblem, ObjectivePolicy, ResourceLimits
from .model.validation import model_issues


def _rule(rng: random.Random, downstream: str, others: List[str]) -> Dict[str, Any]:
    k = rng.randint(1, min(3, len(others)))
    hard = rng.sample(others, k)
    soft = [s for s in rng.sample(others, min(len(others), 1)) if s not in hard]
    kind = rng.choice(["AND", "AND", "OR", "THRESHOLD", "GROUP"])
    rule: Dict[str, Any] = {"downstream": downstream, "rule_type": kind, "semantics": "RIPPLECUT-MODELED",
                            "inputs": [{"service": s, "strength": "hard", "topology": "RIPPLECUT-MODELED"}
                                       for s in hard] +
                                      [{"service": s, "strength": "soft", "topology": "RIPPLECUT-MODELED"}
                                       for s in soft]}
    if kind == "THRESHOLD":
        rule["threshold"] = rng.randint(1, k)
    if kind == "GROUP":
        cut = rng.randint(1, k)
        groups = [hard[:cut], hard[cut:]] if cut < k else [hard]
        rule["groups"] = [{"members": g, "q": rng.randint(1, len(g))} for g in groups if g]
    return rule


DEFAULT_COSTS = (0, 1, 1, 2, 2, 3, 4, 5, 7)


def random_raw_model(seed: int, n_services: int = 8, m_actions: int = 6, cycle_prob: float = 0.3,
                     allow_joint: bool = True, crit_frac: float = 0.4,
                     cost_choices: Tuple[int, ...] = DEFAULT_COSTS,
                     ensure_up: Tuple[str, ...] = ()) -> Dict[str, Any]:
    rng = random.Random(seed)
    services = [f"s{i:02d}" for i in range(n_services)]
    rules = []
    for i, s in enumerate(services):
        if i == 0 or rng.random() < 0.25:
            continue                                     # a root: R_v = 1
        pool = services[:i] + (services[i + 1:] if rng.random() < cycle_prob else [])
        rules.append(_rule(rng, s, pool))
    n_crit = max(1, round(crit_frac * n_services))
    critical = sorted(rng.sample(services, n_crit))
    actions = []
    for j in range(m_actions):
        aid = f"a{j:02d}"
        up = rng.sample(services, rng.randint(1, 2))
        down = [s for s in rng.sample(services, 1) if s not in up] if rng.random() < 0.12 else []
        pre = [{"service": up[0], "state": "DOWN"}] if rng.random() < 0.3 else []
        actions.append({"id": aid, "name": aid, "cost": rng.choice(list(cost_choices)),
                        "targets": sorted(set(up) | set(down)), "preconditions": pre,
                        "effect": {"set_up": sorted(up), "set_down": sorted(down)}})
    for f in ensure_up:                                   # plant at least one restoring action per failure
        if not any(f in a["effect"]["set_up"] for a in actions):
            a = rng.choice(actions)
            a["effect"]["set_up"] = sorted(set(a["effect"]["set_up"]) | {f})
            a["effect"]["set_down"] = [d for d in a["effect"]["set_down"] if d != f]
            a["targets"] = sorted(set(a["effect"]["set_up"]) | set(a["effect"]["set_down"]))
    ids = [a["id"] for a in actions]
    conflicts: Dict[frozenset, Dict[str, Any]] = {}
    for a, b in combinations(ids, 2):
        if rng.random() < 0.08:
            conflicts[frozenset((a, b))] = {"actions": [a, b], "resolution": "INVALID_COMBINATION", "rationale": "random"}
    if allow_joint and len(ids) >= 2 and rng.random() < 0.5:
        a, b = rng.sample(ids, 2)
        key = frozenset((a, b))
        if key not in conflicts:
            conflicts[key] = {"actions": [a, b], "resolution": "DEFINED_JOINT_EFFECT", "rationale": "random",
                              "joint_effect": {"set_up": [rng.choice(services)], "set_down": []}}
    raw = {
        "model": {"id": f"random_seed_{seed}", "name": f"random model seed={seed}", "seed": seed},
        "services": [{"id": s, "role": "business"} for s in services],
        "dependency_rules": rules,
        "criticality": {"critical": critical, "non_critical": sorted(set(services) - set(critical)),
                        "infrastructure": []},
        "interventions": actions,
        "action_conflicts": list(conflicts.values()),
    }
    # Any undeclared opposite-direction composition must be declared explicitly: make it INVALID.
    for _ in range(10):
        bundle_issues = _issues(raw)
        if not bundle_issues:
            break
        _declare_clashes(raw)
    return raw


def _issues(raw: Dict[str, Any]) -> List[str]:
    try:
        b = parse_model(raw)
    except Exception as exc:  # noqa: BLE001 - validation issues are reported via the exception
        return [str(exc)]
    return model_issues(b.system, b.actions)


def _declare_clashes(raw: Dict[str, Any]) -> None:
    acts = {a["id"]: a for a in raw["interventions"]}
    declared = {frozenset(c["actions"]) for c in raw["action_conflicts"]}
    joint = [c for c in raw["action_conflicts"] if c["resolution"] == "DEFINED_JOINT_EFFECT"]

    def clash(e1, e2) -> bool:
        return bool(set(e1["set_up"]) & set(e2["set_down"]) or set(e1["set_down"]) & set(e2["set_up"]))

    for a, b in combinations(sorted(acts), 2):
        if frozenset((a, b)) not in declared and clash(acts[a]["effect"], acts[b]["effect"]):
            raw["action_conflicts"].append({"actions": [a, b], "resolution": "INVALID_COMBINATION",
                                            "rationale": "opposite effects"})
            declared.add(frozenset((a, b)))
    for c in joint:
        pair = c["actions"]
        for other in sorted(acts):
            if other in pair:
                continue
            if clash(c["joint_effect"], acts[other]["effect"]):
                for x in pair:
                    if frozenset((x, other)) not in declared:
                        raw["action_conflicts"].append({"actions": [x, other], "resolution": "INVALID_COMBINATION",
                                                        "rationale": "clashes with a joint effect"})
                        declared.add(frozenset((x, other)))


def random_problem(seed: int, n_services: int = 8, m_actions: int = 6, n_failures: Tuple[int, int] = (1, 3),
                   limits: ResourceLimits = ResourceLimits(timeout_seconds=60, max_evaluations=10_000_000),
                   plant_restorers: bool = False, **kw: Any) -> Tuple[ContainmentProblem, ModelBundle]:
    rng = random.Random(seed * 7919 + 13)
    failed = rng.sample([f"s{i:02d}" for i in range(n_services)], rng.randint(*n_failures))
    raw = random_raw_model(seed, n_services, m_actions, ensure_up=tuple(sorted(failed)) if plant_restorers else (),
                           **kw)
    bundle = parse_model(raw)
    x0 = bundle.system.state_with_failures(failed)
    problem = ContainmentProblem(problem_id=f"random-{seed}", system=bundle.system, actions=bundle.actions,
                                 initial_state=x0, objective=ObjectivePolicy(), resource_limits=limits,
                                 metadata={"seed": seed, "failed": sorted(failed)})
    return problem, bundle
