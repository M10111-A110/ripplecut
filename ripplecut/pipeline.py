"""End-to-end RippleCut pipeline (master spec §5, §92; PDF §1.3, §13.4).

    incident (structured | text -> parser | replay fixture -> state estimator)
      -> x(0) -> uncontrolled cascade (Phi)
      -> ContainmentProblem -> SolverOrchestrator (registry + policy + guard + validator)
      -> contained cascade of the accepted plan -> deterministic explanation
      -> PENDING_HUMAN_APPROVAL

The pipeline is solver-agnostic: it receives a SolverRegistry and a
SolverPolicy and never names an algorithm. It never crashes on bad input:
input problems become explicit statuses in the run report.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import __version__
from .engine.dependency import rules_summary
from .engine.objective import evaluate_plan
from .engine.simulator import simulate
from .errors import ErrorCode, RippleCutError
from .evidence.sources import ReplayFixtureSource
from .evidence.state_estimator import StateEstimator
from .explain.explainer import explain
from .explain.llm_explainer import llm_explanation
from .incident.llm import LLMIncidentParser, LLMUnavailable
from .incident.parser import OK, ParseResult, RuleBasedParser
from .incident.schema import StructuredIncident, validate_incident
from .logs import EventRecorder, get_logger, log_event
from .model.loader import REPO_ROOT, ModelBundle, load_model, resolve_config_path
from .model.schema import ContainmentProblem, State, cost_to_json
from .solvers.builtin import build_default_registry
from .solvers.fault_injection import FAULT_MODES, RESILIENCE_LABEL, FaultInjectingSolver
from .solvers.orchestrator import SolverOrchestrator
from .solvers.policy import SolverPolicy, load_policy
from .solvers.registry import SolverRegistry
from .validation.validator import SafetyValidator

DEFAULT_SCENARIOS_PATH = resolve_config_path("scenarios.json")
CONTROLLED_LABEL = "Controlled RippleCut Scenario"
APPROVAL_PENDING = "PENDING_HUMAN_APPROVAL"
VOLATILE_KEYS = frozenset({"run_id", "runtime_seconds", "guard_runtime_seconds", "events", "timing", "created_at",
                           "traceback"})


@dataclass(frozen=True)
class FaultSpec:
    """Solver failure simulation / resilience test. ``solver=None`` targets the first solver in the policy order."""
    mode: str
    solver: Optional[str] = None
    timeout_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.mode not in FAULT_MODES:
            raise RippleCutError(f"unknown fault mode {self.mode!r}; choose from {list(FAULT_MODES)}",
                                 ErrorCode.CONFIG_ERROR)


def load_scenarios(path: Optional[Path], bundle: ModelBundle) -> List[Dict[str, Any]]:
    """Load and validate demo scenarios (fail early, master spec §102)."""
    p = Path(path) if path else DEFAULT_SCENARIOS_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RippleCutError(f"cannot load scenarios {p}: {exc}", ErrorCode.CONFIG_ERROR) from None
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise RippleCutError("scenarios file must contain a non-empty 'scenarios' list", ErrorCode.CONFIG_ERROR)
    seen = set()
    for s in scenarios:
        sid = s.get("id")
        if not isinstance(sid, str) or sid in seen:
            raise RippleCutError(f"scenario id missing or duplicated: {sid!r}", ErrorCode.CONFIG_ERROR)
        seen.add(sid)
        if ("incident" in s) == ("fixture" in s):
            raise RippleCutError(f"scenario {sid}: define exactly one of 'incident' or 'fixture'", ErrorCode.CONFIG_ERROR)
        if "incident" in s:
            validate_incident(s["incident"], bundle.system)               # raises INVALID_SERVICE etc.
        else:
            fx = REPO_ROOT / s["fixture"]
            if not fx.exists():
                raise RippleCutError(f"scenario {sid}: fixture {fx} not found", ErrorCode.CONFIG_ERROR)
        if "RCAEval" in s.get("label", "") and "NOT RCAEval" not in s.get("label", ""):
            raise RippleCutError(f"scenario {sid}: no RCAEval case was inspected; it must not be labeled as one",
                                 ErrorCode.CONFIG_ERROR)
    return scenarios


def deterministic_view(obj: Any) -> Any:
    """Report without wall-clock/volatile fields: equal inputs must give equal views (master spec §46)."""
    if isinstance(obj, Mapping):
        return {k: deterministic_view(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, (list, tuple)):
        return [deterministic_view(v) for v in obj]
    return obj


def fingerprint(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(deterministic_view(report), sort_keys=True, default=str).encode()).hexdigest()


class RippleCutApp:
    def __init__(self, model_path: Optional[Path] = None, policy_path: Optional[Path] = None,
                 scenarios_path: Optional[Path] = None, registry: Optional[SolverRegistry] = None,
                 policy: Optional[SolverPolicy] = None, llm_client: Any = None) -> None:
        self.bundle = load_model(model_path)
        self.policy = policy or load_policy(policy_path)
        self.registry = registry or build_default_registry()
        self.policy.validate_against(self.registry)
        self.scenarios = load_scenarios(scenarios_path, self.bundle)
        self.estimator = StateEstimator(self.bundle.state_estimation)
        self.rule_parser = RuleBasedParser(self.bundle.system, self.bundle.raw)
        self.llm_client = llm_client

    # ---- public entry points ---------------------------------------------------
    def scenario(self, scenario_id: str) -> Dict[str, Any]:
        for s in self.scenarios:
            if s["id"] == scenario_id:
                return s
        raise RippleCutError(f"unknown scenario '{scenario_id}'; available: {[s['id'] for s in self.scenarios]}",
                             ErrorCode.CONFIG_ERROR)

    def run_scenario(self, scenario_id: str, *, via_text: bool = False, fault: Optional[FaultSpec] = None,
                     use_llm: bool = False) -> Dict[str, Any]:
        s = self.scenario(scenario_id)
        meta = {"id": s["id"], "title": s.get("title", ""), "label": s.get("label", CONTROLLED_LABEL)}
        if "fixture" in s:
            return self.run_fixture(REPO_ROOT / s["fixture"], fault=fault, scenario=meta)
        if via_text and s.get("example_text"):
            return self.run_text(s["example_text"], fault=fault, scenario=meta, use_llm=use_llm)
        return self.run_incident(s["incident"], fault=fault, scenario=meta)

    def run_incident(self, incident: Mapping[str, Any], *, fault: Optional[FaultSpec] = None,
                     scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with EventRecorder() as rec:
            report = self._base(scenario, {"kind": "structured", "incident": dict(incident)}, fault)
            try:
                inc = validate_incident(incident, self.bundle.system)
            except RippleCutError as exc:
                return self._input_error(report, exc, rec)
            return self._core(report, inc, inc.initial_state(self.bundle.system), fault, rec)

    def parse_text(self, text: str, use_llm: bool = False) -> ParseResult:
        if use_llm and self.llm_client is not None:
            try:
                return LLMIncidentParser(self.bundle.system, self.llm_client).parse(text)
            except LLMUnavailable as exc:
                result = self.rule_parser.parse(text)
                result.notes.append(f"LLM parser unavailable ({exc.message}); deterministic parser used")
                return result
        result = self.rule_parser.parse(text)
        if use_llm:
            result.notes.append("no LLM client configured (offline mode); deterministic parser used")
        return result

    def run_text(self, text: str, *, fault: Optional[FaultSpec] = None, scenario: Optional[Dict[str, Any]] = None,
                 use_llm: bool = False) -> Dict[str, Any]:
        with EventRecorder() as rec:
            report = self._base(scenario, {"kind": "text", "text": text}, fault)
            t0 = time.perf_counter()
            parsed = self.parse_text(text, use_llm=use_llm)
            report["timing"]["parse_seconds"] = round(time.perf_counter() - t0, 6)
            report["parse"] = parsed.to_dict()
            log_event(get_logger(), "incident", source=parsed.parser, status=parsed.status,
                      structured=parsed.incident.canonical() if parsed.incident else None)
            if not parsed.ok or parsed.incident is None:
                report["status"] = {"AMBIGUOUS": "INCIDENT_AMBIGUOUS", "UNKNOWN_SERVICE": "INVALID_SERVICE",
                                    "NO_SERVICE_FOUND": "INCIDENT_AMBIGUOUS",
                                    "LLM_PARSE_ERROR": "LLM_PARSE_ERROR"}.get(parsed.status, parsed.status)
                report["error"] = {"code": report["status"], "message": parsed.message}
                report["events"] = rec.events
                return report
            return self._core(report, parsed.incident, parsed.incident.initial_state(self.bundle.system), fault, rec,
                              use_llm=use_llm)

    def run_fixture(self, path: Path, *, fault: Optional[FaultSpec] = None,
                    scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with EventRecorder() as rec:
            report = self._base(scenario, {"kind": "replay_fixture", "path": str(Path(path).relative_to(REPO_ROOT))
                                           if Path(path).resolve().is_relative_to(REPO_ROOT) else str(path)}, fault)
            try:
                bundle = ReplayFixtureSource(Path(path)).load()
                est = self.estimator.estimate(self.bundle.system, bundle.observations, bundle.label)
            except RippleCutError as exc:
                return self._input_error(report, exc, rec)
            report["evidence"] = {**bundle.to_dict(), "estimated_state": est.to_dict()}
            desc = est.descriptive()
            inc = StructuredIncident(failed_services=tuple(sorted(s for s, v in desc.items() if v == "DOWN")),
                                     degraded_services=tuple(sorted(s for s, v in desc.items() if v == "DEGRADED")),
                                     scenario_id=(scenario or {}).get("id"), source="replay_fixture",
                                     notes=(f"state estimated from {bundle.label}",))
            return self._core(report, inc, est.formal_state(self.bundle.system), fault, rec)

    # ---- internals ----------------------------------------------------------------
    def _base(self, scenario: Optional[Dict[str, Any]], inp: Dict[str, Any], fault: Optional[FaultSpec]) -> Dict[str, Any]:
        system = self.bundle.system
        return {
            "run_id": uuid.uuid4().hex[:12],
            "ripplecut_version": __version__,
            "label": (scenario or {}).get("label", f"{CONTROLLED_LABEL} (ad-hoc incident)"),
            "scenario": scenario,
            "model": {"id": system.model_id, "name": system.name,
                      "source_commit_inspected": system.metadata.get("source_commit_inspected")},
            "input": inp,
            "resilience_test": ({"label": RESILIENCE_LABEL, "mode": fault.mode,
                                 "target_solver": fault.solver or self.policy.execution_order()[0],
                                 "timeout_seconds": fault.timeout_seconds} if fault else None),
            "status": None,
            "timing": {},
        }

    def _input_error(self, report: Dict[str, Any], exc: RippleCutError, rec: EventRecorder) -> Dict[str, Any]:
        report["status"] = exc.code.value
        report["error"] = exc.to_dict()
        log_event(get_logger(), "input_rejected", code=exc.code.value, message=exc.message)
        report["events"] = rec.events
        return report

    def _registry_and_policy(self, fault: Optional[FaultSpec]):
        if fault is None:
            return self.registry, self.policy
        target = fault.solver or self.policy.execution_order()[0]
        registry = self.registry.with_override(FaultInjectingSolver(self.registry.get(target), fault.mode))
        policy = self.policy.with_timeout(fault.timeout_seconds) if fault.mode == "timeout" else self.policy
        return registry, policy

    def _core(self, report: Dict[str, Any], inc: StructuredIncident, x0: State, fault: Optional[FaultSpec],
              rec: EventRecorder, use_llm: bool = False) -> Dict[str, Any]:
        system, actions, log = self.bundle.system, self.bundle.actions, get_logger()
        report["incident"] = inc.to_dict()
        report["initial_state"] = system.state_to_mapping(x0)
        problem_id = (inc.scenario_id or "adhoc") + "-" + hashlib.sha256(
            json.dumps(system.state_to_mapping(x0), sort_keys=True).encode()).hexdigest()[:8]
        log_event(log, "initial_state", scenario_id=inc.scenario_id, problem_id=problem_id,
                  down=list(system.down_services(x0)))

        t0 = time.perf_counter()
        uncontrolled = simulate(system, x0)
        report["timing"]["uncontrolled_cascade_seconds"] = round(time.perf_counter() - t0, 6)
        report["uncontrolled_cascade"] = uncontrolled.to_dict(system)
        for r in uncontrolled.history[1:]:
            log_event(log, "cascade_round", round=r.round, newly_failed=list(r.newly_failed))
        if not uncontrolled.fixed_point:
            report["status"] = ErrorCode.SIMULATION_ERROR.value
            report["error"] = {"code": "SIMULATION_ERROR", "message": uncontrolled.termination_reason}
            report["events"] = rec.events
            return report

        registry, policy = self._registry_and_policy(fault)
        problem = ContainmentProblem(problem_id=problem_id, system=system, actions=actions, initial_state=x0,
                                     objective=self.bundle.objective, resource_limits=policy.limits,
                                     metadata={"scenario_id": inc.scenario_id})
        report["problem"] = {"problem_id": problem_id, "m": problem.m, "candidates": 2 ** problem.m,
                             "C_min": system.c_min, "objective": list(problem.objective.order),
                             "tie_breaker": problem.objective.final_tie_breaker}
        t0 = time.perf_counter()
        orchestrator = SolverOrchestrator(registry, policy, validator=SafetyValidator(
            policy.accept_uncertified_infeasibility_from_exact_solver))
        orch = orchestrator.solve(problem)
        report["timing"]["planning_seconds"] = round(time.perf_counter() - t0, 6)
        report["planner"] = {"available_solvers": registry.describe(), "policy": dict(policy.to_dict()),
                             **orch.to_dict(problem)}
        report["status"] = orch.status

        contained = None
        if orch.status == "SUCCESS":
            contained = evaluate_plan(problem, orch.plan or ())
            report["contained_cascade"] = {"plan": list(orch.plan or ()),
                                           "post_intervention_state": system.state_to_mapping(
                                               contained.post_intervention_state),
                                           **contained.cascade.to_dict(system)}
            report["result"] = {"plan": list(orch.plan or ()), "objective": contained.objective.to_dict(),
                                "optimality_status": orch.optimality_status.value,
                                "validation_status": "VALID" if orch.validation and orch.validation.valid else "INVALID",
                                "selected_solver": orch.selected_solver, "fallback_count": orch.fallback_count}
        elif orch.status == "NO_FEASIBLE_PLAN":
            report["result"] = {"plan": None, "optimality_status": orch.optimality_status.value,
                                "validation_status": "VALID", "selected_solver": orch.selected_solver,
                                "fallback_count": orch.fallback_count,
                                "infeasibility_certified": orch.validation.infeasibility_certified,
                                "infeasibility_basis": orch.optimality_basis}
        else:
            report["result"] = {"plan": None, "optimality_status": "UNKNOWN", "validation_status": "NONE_ACCEPTED",
                                "selected_solver": None, "fallback_count": orch.fallback_count}

        t0 = time.perf_counter()
        infeas = orchestrator.validator.infeasibility_certificate(problem) if orch.status == "NO_FEASIBLE_PLAN" else None
        report["explanation"] = explain(problem, uncontrolled, orch, contained, infeas)
        if use_llm and self.llm_client is not None:
            report["llm_explanation"] = llm_explanation(self.llm_client, report["explanation"], system, actions)
        report["timing"]["explanation_seconds"] = round(time.perf_counter() - t0, 6)
        report["approval"] = {"status": APPROVAL_PENDING if orch.status == "SUCCESS" else "NOTHING_TO_APPROVE",
                              "note": "RippleCut only recommends a modeled plan. Approval in the demo is simulated; "
                                      "no real-world action is executed."}
        log_event(log, "objective_values", objective=report["result"].get("objective"),
                  optimality=report["result"]["optimality_status"])
        report["events"] = rec.events
        return report

    # ---- descriptive views -----------------------------------------------------------
    def model_view(self) -> Dict[str, Any]:
        system, actions = self.bundle.system, self.bundle.actions
        edges = []
        for r in system.rules.values():
            for inp in r.inputs:
                edges.append({"from": r.downstream, "to": inp.service, "strength": inp.strength.value,
                              "topology": inp.topology, "evidence": inp.evidence})
        return {
            "model": {"id": system.model_id, "name": system.name, **{k: v for k, v in system.metadata.items()
                                                                     if k in ("reference_system", "source_repository",
                                                                              "source_commit_inspected", "labeling")}},
            "services": [{"id": s.id, "role": s.role.value, "description": s.description,
                          "critical": s.id in system.critical,
                          "class": "critical" if s.id in system.critical else
                          ("infrastructure" if s.id in system.infrastructure else "non_critical")}
                         for s in system.services],
            "edges": edges, "rules": dict(rules_summary(system)),
            "rule_semantics_label": "RIPPLECUT-MODELED RULE", "topology_label": "SOURCE-BACKED TOPOLOGY",
            "c_min": system.c_min,
            "actions": [{"id": a.id, "name": a.name, "cost": cost_to_json(a.cost), "description": a.description,
                         "set_up": sorted(a.effect.set_up), "set_down": sorted(a.effect.set_down),
                         "preconditions": [{"service": p.service, "state": "UP" if p.state else "DOWN"}
                                           for p in a.preconditions], "conflicts": list(a.conflicts),
                         "label": a.metadata.get("label", "RIPPLECUT-MODELED")} for a in actions.interventions],
            "conflicts": [{"actions": sorted(c.actions), "resolution": c.resolution.value, "rationale": c.rationale}
                          for c in actions.conflicts.values()],
            "layout": dict(self.bundle.presentation.get("layout", {})),
            "solvers": self.registry.describe(), "policy": dict(self.policy.to_dict()),
            "scenarios": [{"id": s["id"], "title": s.get("title", ""), "label": s.get("label", CONTROLLED_LABEL),
                           "example_text": s.get("example_text"), "kind": "fixture" if "fixture" in s else "incident"}
                          for s in self.scenarios],
            "fault_modes": list(FAULT_MODES), "resilience_label": RESILIENCE_LABEL,
        }


def validate_config(app: Optional[RippleCutApp] = None) -> Dict[str, Any]:
    """All configuration checks of master spec §102 (construction already fails early on any error)."""
    app = app or RippleCutApp()
    checks = [
        ("model parses and validates (services, rules, criticality, actions, conflicts, costs)", True),
        ("solver policy references only registered solvers; fallback order uses enabled solvers", True),
        ("scenario incidents reference valid services; fixtures exist", True),
    ]
    for s in app.scenarios:
        if "fixture" in s:
            b = ReplayFixtureSource(REPO_ROOT / s["fixture"]).load()
            app.estimator.estimate(app.bundle.system, b.observations, b.label)
            checks.append((f"fixture of scenario {s['id']} maps to a complete state", True))
    return {"valid": True, "checks": [{"check": c, "passed": p} for c, p in checks],
            "model": app.bundle.system.model_id, "services": len(app.bundle.system.services),
            "hard_dependencies": sum(len(r.hard_upstream) for r in app.bundle.system.rules.values()),
            "soft_dependencies": sum(len(r.soft_upstream) for r in app.bundle.system.rules.values()),
            "actions": len(app.bundle.actions), "execution_order": list(app.policy.execution_order())}
