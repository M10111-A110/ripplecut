"""Shared fixtures and tiny hand-built models for the RippleCut test suite."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripplecut.model.loader import load_model, parse_model  # noqa: E402
from ripplecut.model.schema import ContainmentProblem, ObjectivePolicy, ResourceLimits  # noqa: E402
from ripplecut.pipeline import RippleCutApp  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return load_model()


@pytest.fixture(scope="session")
def app():
    return RippleCutApp()


def raw_model(services: Sequence[str], rules: Iterable[Dict[str, Any]] = (), critical: Iterable[str] = (),
              actions: Iterable[Dict[str, Any]] = (), conflicts: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    crit = sorted(set(critical))
    return {
        "model": {"id": "test", "name": "test model"},
        "services": [{"id": s, "role": "business"} for s in services],
        "dependency_rules": [dict(r, semantics=r.get("semantics", "RIPPLECUT-MODELED")) for r in rules],
        "criticality": {"critical": crit, "non_critical": sorted(set(services) - set(crit)), "infrastructure": []},
        "interventions": list(actions),
        "action_conflicts": list(conflicts),
    }


def hard(*names: str) -> List[Dict[str, Any]]:
    return [{"service": n, "strength": "hard", "topology": "RIPPLECUT-MODELED"} for n in names]


def soft(*names: str) -> List[Dict[str, Any]]:
    return [{"service": n, "strength": "soft", "topology": "RIPPLECUT-MODELED"} for n in names]


def rule(downstream: str, rule_type: str = "AND", inputs: Optional[List[Dict[str, Any]]] = None, **kw: Any):
    return {"downstream": downstream, "rule_type": rule_type, "inputs": inputs or [], **kw}


def action(aid: str, cost: Any, up: Iterable[str] = (), down: Iterable[str] = (),
           pre: Iterable[tuple] = ()) -> Dict[str, Any]:
    up, down = sorted(up), sorted(down)
    return {"id": aid, "name": aid, "cost": cost, "targets": sorted(set(up) | set(down)),
            "preconditions": [{"service": s, "state": st} for s, st in pre],
            "effect": {"set_up": up, "set_down": down}}


def build(**kw: Any):
    return parse_model(raw_model(**kw))


def make_problem(b, failed: Iterable[str] = (), state: Optional[Dict[str, int]] = None,
                 limits: ResourceLimits = ResourceLimits(timeout_seconds=10, max_evaluations=1_000_000),
                 pid: str = "test") -> ContainmentProblem:
    x0 = b.system.state_from_mapping(state) if state is not None else b.system.state_with_failures(failed)
    return ContainmentProblem(problem_id=pid, system=b.system, actions=b.actions, initial_state=x0,
                              objective=ObjectivePolicy(), resource_limits=limits)
