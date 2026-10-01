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
from typing import Any, Dict, List, Mapping, Tuple

from ..errors import DataAdapterError
from ..model.schema import State, SystemModel

UP, DEGRADED, DOWN = "UP", "DEGRADED", "DOWN"
FIELDS = ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")
THRESHOLD_KEYS = ("down_error_rate_gte", "degraded_error_rate_gte", "degraded_latency_p95_ms_gte",
                  "degraded_cpu_utilization_gte")


@dataclass(frozen=True)
class Observation:
    service: str
    latency_p95_ms: float
    error_rate: float
    request_rate: float
    cpu_utilization: float


@dataclass(frozen=True)
class ServiceState:
    service_id: str
    state: str                 # UP | DEGRADED | DOWN (descriptive)
    formal: int                # 1 iff UP
    observation: Observation
    reason: str
    source: str


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
