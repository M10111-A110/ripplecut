"""Real RCAEval Adapter and Evaluation Boundary (Phase 4, master spec §4.1-§4.5).

This module bridges raw RCAEval case directories to RippleCut:
    RCAEvalRawCase
          |
          v
    RCAEvalAdapter
          |
          v
    NormalizedObservations
          |
          v
    StateEstimator
          |
          v
    RippleCutState

Key Guarantees:
1. RCAEval-specific column and metric schemas never leak into the deterministic core.
2. Missing metrics are handled explicitly via SignalStatus (AVAILABLE, MISSING, UNSUPPORTED, AMBIGUOUS).
   Missing telemetry is NEVER silently assumed to be DOWN.
3. RCA evaluation (ground truth root-cause localization) is strictly separated from
   containment evaluation (plan validity, cost, critical preservation, optimality).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..errors import DataAdapterError
from ..model.schema import State, SystemModel
from .state_estimator import DOWN, DEGRADED, UP, Observation, StateEstimator


class SignalStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


class EvidenceClassification(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class DataProvenance:
    source_dataset: str
    source_repository: str
    dataset_revision: str
    case_id: str
    benchmark: str
    ground_truth_service: str
    fault_type: str
    inject_time: float
    original_metric_count: Optional[int] = None
    original_timestep_count: Optional[int] = None
    source_artifact: str = ""
    transformation_version: str = "1.0"
    data_status: str = "SYNTHETIC_DERIVED"


@dataclass(frozen=True)
class RCAEvalCaseMetadata:
    case_id: str
    case_dir: str
    benchmark: str
    ground_truth_service: str
    fault_type: str
    instance: int
    inject_time: float
    source: str = "RCAEval"
    data_status: str = "OFFICIAL_RCAEVAL"
    source_reference: str = "https://github.com/phamquiluan/RCAEval"
    retrieval_version: str = "1.0"
    inject_time_validated: bool = False


@dataclass(frozen=True)
class ServiceSignal:
    name: str
    status: SignalStatus
    value: Optional[float] = None
    baseline_value: Optional[float] = None
    detail: str = ""


@dataclass(frozen=True)
class NormalizedServiceObservation:
    service: str
    latency_p95_ms: Optional[float]
    error_rate: Optional[float]
    request_rate: Optional[float]
    cpu_utilization: Optional[float]
    signals: Mapping[str, ServiceSignal]
    missing_signals: Tuple[str, ...]

    def to_observation_dict(self) -> Dict[str, float]:
        if self.missing_signals:
            raise DataAdapterError(f"Cannot use to_observation_dict when signals are missing: {self.missing_signals}")
        return {
            "latency_p95_ms": float(self.latency_p95_ms) if self.latency_p95_ms is not None else 0.0,
            "error_rate": float(self.error_rate) if self.error_rate is not None else 0.0,
            "request_rate": float(self.request_rate) if self.request_rate is not None else 0.0,
            "cpu_utilization": float(self.cpu_utilization) if self.cpu_utilization is not None else 0.0,
        }

    def to_partial_observation_dict(self) -> Dict[str, Any]:
        return {
            "latency_p95_ms": self.latency_p95_ms,
            "error_rate": self.error_rate,
            "request_rate": self.request_rate,
            "cpu_utilization": self.cpu_utilization,
            "signals": self.signals,
            "missing_signals": self.missing_signals,
        }


@dataclass(frozen=True)
class RCAEvalCaseData:
    metadata: RCAEvalCaseMetadata
    raw_columns: Tuple[str, ...]
    timestamps: Tuple[float, ...]
    pre_window_count: int
    post_window_count: int
    normalized_observations: Mapping[str, NormalizedServiceObservation]


@dataclass(frozen=True)
class RCAEvaluationReport:
    """Evaluation of Root Cause Localization against RCAEval ground truth."""
    case_id: str
    benchmark: str
    ground_truth_service: str
    fault_type: str
    inject_time: float
    localized_services: Tuple[str, ...]
    root_cause_detected: bool
    state_classification: Mapping[str, str]
    signals_available: Mapping[str, List[str]]
    signals_missing: Mapping[str, List[str]]
    metric_name: str = "ground_truth_presence"
    metric_note: str = "This P0 metric checks whether the authoritative RCAEval root-cause service appears among services classified as anomalous. It is not a full RCAEval ranking metric."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "benchmark": self.benchmark,
            "ground_truth_service": self.ground_truth_service,
            "fault_type": self.fault_type,
            "inject_time": self.inject_time,
            "localized_services": list(self.localized_services),
            "root_cause_detected": self.root_cause_detected,
            "state_classification": dict(self.state_classification),
            "signals_available": {k: list(v) for k, v in self.signals_available.items()},
            "signals_missing": {k: list(v) for k, v in self.signals_missing.items()},
            "metric_name": self.metric_name,
            "metric_note": self.metric_note,
        }


def parse_case_directory_name(dir_name: str) -> Tuple[str, str, str, int]:
    """Parse {benchmark}_{service}_{fault}_{instance} naming convention."""
    parts = dir_name.strip("/\\").split("_")
    if len(parts) >= 4:
        benchmark = parts[0]
        service = parts[1]
        fault = parts[2]
        try:
            instance = int(parts[3])
        except ValueError:
            instance = 1
        return benchmark, service, fault, instance

    # Fallback if delimiter varies
    return "rcaeval", dir_name, "unknown", 1


def validate_directory_against_metadata(dir_name: str, metadata: RCAEvalCaseMetadata) -> None:
    parsed_benchmark, parsed_service, parsed_fault, parsed_instance = parse_case_directory_name(dir_name)
    errors = []
    if parsed_service != metadata.ground_truth_service:
        errors.append(f"directory says service='{parsed_service}' but metadata says '{metadata.ground_truth_service}'")
    if parsed_fault != metadata.fault_type:
        errors.append(f"directory says fault='{parsed_fault}' but metadata says '{metadata.fault_type}'")
    if errors:
        raise DataAdapterError(f"DATA_ADAPTER_ERROR: case directory name disagrees with authoritative metadata: {'; '.join(errors)}")


class RCAEvalAdapter:
    """Adapts raw RCAEval directory into normalized observations with provenance."""

    def __init__(self, system: SystemModel, estimator_config: Mapping[str, Any]) -> None:
        self.system = system
        self.estimator_config = estimator_config

    def load_case(self, case_dir: Path) -> RCAEvalCaseData:
        case_dir = Path(case_dir)
        if not case_dir.is_dir():
            raise DataAdapterError(f"RCAEval case directory not found: {case_dir}")

        metadata_file = case_dir / "metadata.json"
        if metadata_file.exists():
            try:
                meta_json = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataAdapterError(f"Failed to parse metadata.json in {case_dir}: {exc}") from None
            metadata = RCAEvalCaseMetadata(
                case_id=str(meta_json.get("case_id", case_dir.name)),
                case_dir=str(case_dir),
                benchmark=str(meta_json.get("benchmark", "re1ob")),
                ground_truth_service=str(meta_json.get("ground_truth_service", "")),
                fault_type=str(meta_json.get("fault_type", "unknown")),
                instance=int(meta_json.get("instance", 1)),
                inject_time=float(meta_json.get("inject_time", 0.0)),
                source=str(meta_json.get("source", "RCAEval")),
                data_status=str(meta_json.get("data_status", meta_json.get("source", "OFFICIAL_RCAEVAL"))),
                source_reference=str(meta_json.get("source_reference", "https://github.com/phamquiluan/RCAEval")),
                retrieval_version=str(meta_json.get("retrieval_version", "1.0")),
            )
            validate_directory_against_metadata(case_dir.name, metadata)
        else:
            inject_file = case_dir / "inject_time.txt"
            if not inject_file.exists():
                raise DataAdapterError(f"RCAEval case missing inject_time.txt or metadata.json in {case_dir}")

            try:
                inject_time = float(inject_file.read_text(encoding="utf-8").strip())
            except (ValueError, TypeError) as exc:
                raise DataAdapterError(f"Invalid timestamp in {inject_file}: {exc}") from None

            benchmark, service, fault, instance = parse_case_directory_name(case_dir.name)
            metadata = RCAEvalCaseMetadata(
                case_id=case_dir.name,
                case_dir=str(case_dir),
                benchmark=benchmark,
                ground_truth_service=service,
                fault_type=fault,
                instance=instance,
                inject_time=inject_time,
            )

        metrics_parquet = case_dir / "metrics.parquet"
        metrics_json = case_dir / "metrics.json"

        raw_metrics = None
        if metrics_parquet.exists():
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(metrics_parquet)
                raw_metrics = {col: [x if x is not None else 0.0 for x in table[col].to_pylist()] for col in table.column_names}
            except Exception:
                raw_metrics = None

        if raw_metrics is None:
            if not metrics_json.exists():
                raise DataAdapterError(f"RCAEval case missing metrics.json or metrics.parquet in {case_dir}")
            try:
                raw_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataAdapterError(f"Failed to parse metrics.json in {case_dir}: {exc}") from None

        return self._normalize_metrics(raw_metrics, metadata)

    def _normalize_metrics(self, raw_metrics: Any, metadata: RCAEvalCaseMetadata) -> RCAEvalCaseData:
        # Standardize representation into columns: Mapping[col_name, List[float]]
        cols: Dict[str, List[float]] = {}
        if isinstance(raw_metrics, Mapping):
            for k, v in raw_metrics.items():
                if isinstance(v, Sequence) and not isinstance(v, str):
                    cols[k] = [float(x) for x in v]
        elif isinstance(raw_metrics, Sequence):
            # List of row objects
            for row in raw_metrics:
                if isinstance(row, Mapping):
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            cols.setdefault(k, []).append(float(v))

        if not cols:
            raise DataAdapterError(f"No metric series found in metrics.json for {metadata.case_id}")

        # Find timestamp series
        time_key = next((k for k in ("time", "timestamp", "ts") if k in cols), None)
        if time_key is None:
            raise DataAdapterError(f"metrics.json missing 'time' column in {metadata.case_id}")

        times = tuple(cols[time_key])
        inject_time = metadata.inject_time

        pre_mask = [t < inject_time for t in times]
        post_mask = [t >= inject_time for t in times]

        pre_count = sum(pre_mask)
        post_count = sum(post_mask)

        # If all points are after or before, inject_time does not meaningfully intersect the range
        if post_count == 0 or pre_count == 0:
            raise DataAdapterError(f"inject_time {inject_time} does not meaningfully intersect the timestamp range")

        normalized_by_service: Dict[str, NormalizedServiceObservation] = {}

        for sid in self.system.service_ids:
            norm_obs = self._extract_service_signals(sid, cols, pre_mask, post_mask)
            normalized_by_service[sid] = norm_obs

        return RCAEvalCaseData(
            metadata=metadata,
            raw_columns=tuple(sorted(cols.keys())),
            timestamps=times,
            pre_window_count=pre_count,
            post_window_count=post_count,
            normalized_observations=normalized_by_service,
        )

    def _extract_service_signals(
        self,
        service: str,
        cols: Mapping[str, List[float]],
        pre_mask: List[bool],
        post_mask: List[bool],
    ) -> NormalizedServiceObservation:
        signals: Dict[str, ServiceSignal] = {}

        # Signal categories to find: cpu_utilization, error_rate, latency_p95_ms, request_rate
        def find_series(patterns: Sequence[str]) -> Optional[List[float]]:
            # 1. Exact match <service>_<pattern>
            for p in patterns:
                key = f"{service}_{p}"
                if key in cols:
                    return cols[key]
            # 2. Case-insensitive or dot-separated
            for k in cols:
                k_lower = k.lower()
                for p in patterns:
                    if k_lower in (f"{service.lower()}_{p}", f"{service.lower()}.{p}", f"{service.lower()}:{p}"):
                        return cols[k]
            return None

        # 1. CPU
        cpu_series = find_series(["cpu", "cpu_utilization", "cpu_usage", "container_cpu_usage"])
        if cpu_series is not None:
            pre_vals = [cpu_series[i] for i, m in enumerate(pre_mask) if m and i < len(cpu_series)]
            post_vals = [cpu_series[i] for i, m in enumerate(post_mask) if m and i < len(cpu_series)]
            b_val = (sum(pre_vals) / len(pre_vals)) if pre_vals else 0.0
            p_val = (sum(post_vals) / len(post_vals)) if post_vals else b_val
            # If percentage, scale to [0, 1]
            if p_val > 1.0:
                p_val = min(1.0, p_val / 100.0)
            if b_val > 1.0:
                b_val = min(1.0, b_val / 100.0)
            signals["cpu_utilization"] = ServiceSignal("cpu_utilization", SignalStatus.AVAILABLE, p_val, b_val)
        else:
            signals["cpu_utilization"] = ServiceSignal("cpu_utilization", SignalStatus.MISSING, value=None)

        # 2. Error rate
        err_series = find_series(["error_rate", "error", "errors", "error_count"])
        if err_series is not None:
            pre_vals = [err_series[i] for i, m in enumerate(pre_mask) if m and i < len(err_series)]
            post_vals = [err_series[i] for i, m in enumerate(post_mask) if m and i < len(err_series)]
            b_val = (sum(pre_vals) / len(pre_vals)) if pre_vals else 0.0
            p_val = (sum(post_vals) / len(post_vals)) if post_vals else b_val
            if p_val > 1.0:
                p_val = min(1.0, p_val / 100.0)
            if b_val > 1.0:
                b_val = min(1.0, b_val / 100.0)
            signals["error_rate"] = ServiceSignal("error_rate", SignalStatus.AVAILABLE, p_val, b_val)
        else:
            signals["error_rate"] = ServiceSignal("error_rate", SignalStatus.MISSING, value=None)

        # 3. Latency
        lat_series = find_series(["latency", "latency_p95_ms", "latency-90", "latency-50", "latency_p90", "duration", "response_time"])
        if lat_series is not None:
            pre_vals = [lat_series[i] for i, m in enumerate(pre_mask) if m and i < len(lat_series)]
            post_vals = [lat_series[i] for i, m in enumerate(post_mask) if m and i < len(lat_series)]
            b_val = (sum(pre_vals) / len(pre_vals)) if pre_vals else 10.0
            p_val = max(post_vals) if post_vals else b_val
            # If latency values are reported in seconds (e.g. <= 30.0 s), convert to milliseconds for threshold comparison
            if p_val <= 30.0 and b_val <= 10.0:
                p_val = p_val * 1000.0
                b_val = b_val * 1000.0
            signals["latency_p95_ms"] = ServiceSignal("latency_p95_ms", SignalStatus.AVAILABLE, p_val, b_val)
        else:
            signals["latency_p95_ms"] = ServiceSignal("latency_p95_ms", SignalStatus.MISSING, value=None)

        # 4. Request rate / QPS
        qps_series = find_series(["qps", "request_rate", "throughput", "requests", "load", "workload"])
        if qps_series is not None:
            pre_vals = [qps_series[i] for i, m in enumerate(pre_mask) if m and i < len(qps_series)]
            post_vals = [qps_series[i] for i, m in enumerate(post_mask) if m and i < len(qps_series)]
            b_val = (sum(pre_vals) / len(pre_vals)) if pre_vals else 50.0
            p_val = (sum(post_vals) / len(post_vals)) if post_vals else b_val
            signals["request_rate"] = ServiceSignal("request_rate", SignalStatus.AVAILABLE, p_val, b_val)
        else:
            signals["request_rate"] = ServiceSignal("request_rate", SignalStatus.MISSING, value=None)

        missing = tuple(k for k, s in signals.items() if s.status is SignalStatus.MISSING)

        return NormalizedServiceObservation(
            service=service,
            latency_p95_ms=signals["latency_p95_ms"].value,
            error_rate=signals["error_rate"].value,
            request_rate=signals["request_rate"].value,
            cpu_utilization=signals["cpu_utilization"].value,
            signals=signals,
            missing_signals=missing,
        )

    def evaluate_rca(self, case_data: RCAEvalCaseData, classified_states: Mapping[str, str]) -> RCAEvaluationReport:
        """Independently evaluate root cause localization against RCAEval ground truth."""
        meta = case_data.metadata
        gt_service = meta.ground_truth_service

        down_or_degraded = tuple(
            s for s, st in classified_states.items() if st in (DOWN, DEGRADED)
        )
        hit = gt_service in down_or_degraded

        avail: Dict[str, List[str]] = {}
        missing: Dict[str, List[str]] = {}
        for sid, obs in case_data.normalized_observations.items():
            avail[sid] = [k for k, sig in obs.signals.items() if sig.status is SignalStatus.AVAILABLE]
            missing[sid] = list(obs.missing_signals)

        return RCAEvaluationReport(
            case_id=meta.case_id,
            benchmark=meta.benchmark,
            ground_truth_service=gt_service,
            fault_type=meta.fault_type,
            inject_time=meta.inject_time,
            localized_services=down_or_degraded,
            root_cause_detected=hit,
            state_classification=classified_states,
            signals_available=avail,
            signals_missing=missing,
        )
