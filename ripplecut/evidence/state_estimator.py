"""Deterministic state estimation (master spec §52, PDF §3.2).

Observation vector z_v = [latency_p95_ms, error_rate, request_rate, cpu_utilization]
(latency, errorRate, requestRate, resourceSignals). Mapping to a descriptive
state uses thresholds read from the model configuration (``state_estimation``);
they are configuration parameters, not learned or universal values:

    DOWN      if error_rate >= down_error_rate_gte
    DEGRADED  elif error_rate >= degraded_error_rate_gte
              or latency_p95_ms >= degraded_latency_p95_ms_gte
              or cpu_utilization >= degraded_cpu_utilization_gte
    UP        otherwise

Formal mapping (PDF §4.1): x_v = 1 iff UP, so DEGRADED -> 0 and DOWN -> 0.
request_rate is recorded and validated but not used by any threshold.
A service without an observation is an explicit DATA_ADAPTER_ERROR (never
silently assumed UP).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Tuple

from ..errors import DataAdapterError
from ..model.schema import State, SystemModel

UP, DEGRADED, DOWN = "UP", "DEGRADED", "DOWN"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
FIELDS = ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")
THRESHOLD_KEYS = ("down_error_rate_gte", "degraded_error_rate_gte", "degraded_latency_p95_ms_gte",
                  "degraded_cpu_utilization_gte")


class EvidenceClassification(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class Observation:
    service: str
    latency_p95_ms: Optional[float] = None
    error_rate: Optional[float] = None
    request_rate: Optional[float] = None
    cpu_utilization: Optional[float] = None


@dataclass(frozen=True)
class ServiceState:
    service_id: str
    state: str                 # UP | DEGRADED | DOWN (descriptive)
    formal: int                # 1 iff UP
    observation: Observation
    reason: str
    source: str
    evidence: str = "SUFFICIENT"
    available_signals: Tuple[str, ...] = ()
    missing_signals: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EstimatedState:
    services: Tuple[ServiceState, ...]
    thresholds: Mapping[str, float]
    source: str

    def formal_state(self, system: SystemModel) -> State:
        by = {s.service_id: s.formal for s in self.services}
        return system.state_from_mapping(by)

    def descriptive(self) -> Dict[str, str]:
        return {s.service_id: s.state for s in self.services}

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "thresholds": dict(self.thresholds),
                "services": [{"service": s.service_id, "state": s.state, "formal_x": s.formal, "reason": s.reason,
                              "evidence": s.evidence, "available_signals": s.available_signals, "missing_signals": s.missing_signals,
                              "observation": {f: getattr(s.observation, f) for f in FIELDS}}
                             for s in self.services]}


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise DataAdapterError(f"{where} must be a finite nonnegative number (got {value!r})")
    return float(value)


def parse_observation(service: str, raw: Any) -> Observation:
    if not isinstance(raw, Mapping):
        raise DataAdapterError(f"observation for {service} must be an object with fields {list(FIELDS)}")
    missing = [f for f in FIELDS if f not in raw]
    extra = sorted(set(raw) - set(FIELDS))
    if missing or extra:
        raise DataAdapterError(f"observation for {service}: missing {missing}, unexpected {extra}")
    vals = {f: _number(raw[f], f"{service}.{f}") for f in FIELDS}
    for f in ("error_rate", "cpu_utilization"):
        if vals[f] > 1:
            raise DataAdapterError(f"{service}.{f} must be a fraction in [0, 1] (got {vals[f]})")
    return Observation(service=service, **vals)


def parse_partial_observation(service: str, raw: Any, missing_signals: Tuple[str, ...]) -> Observation:
    if not isinstance(raw, Mapping):
        raise DataAdapterError(f"observation for {service} must be an object with fields {list(FIELDS)}")
    vals = {}
    for f in FIELDS:
        if f in missing_signals:
            vals[f] = None
        elif f in raw and raw[f] is not None:
            vals[f] = _number(raw[f], f"{service}.{f}")
            if f in ("error_rate", "cpu_utilization") and vals[f] > 1:
                raise DataAdapterError(f"{service}.{f} must be a fraction in [0, 1] (got {vals[f]})")
        else:
            raise DataAdapterError(f"observation for {service}: missing non-optional signal '{f}'")
    return Observation(service=service, **vals)


class StateEstimator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        missing = [k for k in THRESHOLD_KEYS if k not in config]
        if missing:
            raise DataAdapterError(f"state_estimation config is missing thresholds {missing}")
        self.t = {k: _number(config[k], f"state_estimation.{k}") for k in THRESHOLD_KEYS}
        if not self.t["degraded_error_rate_gte"] <= self.t["down_error_rate_gte"]:
            raise DataAdapterError("degraded_error_rate_gte must not exceed down_error_rate_gte")
        policy = config.get("missing_observation", "ERROR")
        if policy != "ERROR":
            raise DataAdapterError("missing_observation policy must be 'ERROR' (no silent defaults)")

    def classify(self, obs: Observation) -> Tuple[str, str]:
        t = self.t
        if obs.error_rate >= t["down_error_rate_gte"]:
            return DOWN, f"error_rate {obs.error_rate:g} >= {t['down_error_rate_gte']:g}"
        reasons: List[str] = []
        if obs.error_rate >= t["degraded_error_rate_gte"]:
            reasons.append(f"error_rate {obs.error_rate:g} >= {t['degraded_error_rate_gte']:g}")
        if obs.latency_p95_ms >= t["degraded_latency_p95_ms_gte"]:
            reasons.append(f"latency_p95_ms {obs.latency_p95_ms:g} >= {t['degraded_latency_p95_ms_gte']:g}")
        if obs.cpu_utilization >= t["degraded_cpu_utilization_gte"]:
            reasons.append(f"cpu_utilization {obs.cpu_utilization:g} >= {t['degraded_cpu_utilization_gte']:g}")
        if reasons:
            return DEGRADED, "; ".join(reasons)
        return UP, "all signals below configured thresholds"

    def estimate(self, system: SystemModel, observations: Mapping[str, Any], source: str) -> EstimatedState:
        if not isinstance(observations, Mapping):
            raise DataAdapterError("observations must map service id -> observation")
        unknown = sorted(set(observations) - set(system.service_ids))
        if unknown:
            raise DataAdapterError(f"observations reference services outside the model: {unknown}")
        missing = [s for s in system.service_ids if s not in observations]
        if missing:
            raise DataAdapterError(f"missing observation(s) for {missing}; RippleCut never assumes a state")
        out = []
        for sid in system.service_ids:
            obs = parse_observation(sid, observations[sid])
            state, reason = self.classify(obs)
            out.append(ServiceState(sid, state, 1 if state == UP else 0, obs, reason, source))
        return EstimatedState(tuple(out), dict(self.t), source)

    def classify_partial(self, obs: Observation, missing: Tuple[str, ...]) -> Tuple[str, str, str]:
        """Returns (state, reason, evidence_classification)"""
        if not missing:
            state, reason = self.classify(obs)
            return state, reason, EvidenceClassification.SUFFICIENT.value
        
        available = set(FIELDS) - set(missing)
        if not available:
            return UP, "no signals available to evaluate", EvidenceClassification.INSUFFICIENT.value
        
        t = self.t
        if "error_rate" in available and obs.error_rate >= t["down_error_rate_gte"]:
            return DOWN, f"error_rate {obs.error_rate:g} >= {t['down_error_rate_gte']:g}", EvidenceClassification.PARTIAL.value
        
        reasons = []
        if "error_rate" in available and obs.error_rate >= t["degraded_error_rate_gte"]:
            reasons.append(f"error_rate {obs.error_rate:g} >= {t['degraded_error_rate_gte']:g}")
        if "latency_p95_ms" in available and obs.latency_p95_ms >= t["degraded_latency_p95_ms_gte"]:
            reasons.append(f"latency_p95_ms {obs.latency_p95_ms:g} >= {t['degraded_latency_p95_ms_gte']:g}")
        if "cpu_utilization" in available and obs.cpu_utilization >= t["degraded_cpu_utilization_gte"]:
            reasons.append(f"cpu_utilization {obs.cpu_utilization:g} >= {t['degraded_cpu_utilization_gte']:g}")
        
        if reasons:
            return DEGRADED, "; ".join(reasons), EvidenceClassification.PARTIAL.value
        
        skipped = ", ".join(sorted(missing))
        return UP, f"available signals below thresholds (missing: {skipped})", EvidenceClassification.PARTIAL.value

    def estimate_partial(self, system: SystemModel, observations: Mapping[str, Any], source: str, signal_availability: Mapping[str, Mapping[str, str]]) -> EstimatedState:
        """
        Like estimate(), but handles partial evidence.
        
        signal_availability: Mapping[service_id, Mapping[signal_name, SignalStatus]]
        """
        if not isinstance(observations, Mapping):
            raise DataAdapterError("observations must map service id -> observation")
        unknown = sorted(set(observations) - set(system.service_ids))
        if unknown:
            raise DataAdapterError(f"observations reference services outside the model: {unknown}")
        missing_services = [s for s in system.service_ids if s not in observations]
        if missing_services:
            raise DataAdapterError(f"missing observation(s) for {missing_services}; partial mode requires an Observation object even if all signals are MISSING")

        out = []
        for sid in system.service_ids:
            svc_availability = signal_availability.get(sid, {})
            missing_sigs = tuple(f for f in FIELDS if svc_availability.get(f) == "MISSING")
            avail_sigs = tuple(f for f in FIELDS if f not in missing_sigs)
            obs = parse_partial_observation(sid, observations[sid], missing_sigs)

            state, reason, evidence = self.classify_partial(obs, missing_sigs)
            
            if evidence == EvidenceClassification.INSUFFICIENT.value:
                formal = 0
                state = INSUFFICIENT_EVIDENCE
            else:
                formal = 1 if state == UP else 0
                
            out.append(ServiceState(
                service_id=sid,
                state=state,
                formal=formal,
                observation=obs,
                reason=reason,
                source=source,
                evidence=evidence,
                available_signals=avail_sigs,
                missing_signals=missing_sigs
            ))
        return EstimatedState(tuple(out), dict(self.t), source)
