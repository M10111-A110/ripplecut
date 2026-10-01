"""Tests for RCAEval real adapter and evaluation boundary (Phase 4, master spec §4.1-§4.5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ripplecut.errors import DataAdapterError
from ripplecut.evidence.rcaeval_adapter import (
    RCAEvalAdapter,
    SignalStatus,
    parse_case_directory_name,
)
from ripplecut.evidence.sources import RCAEvalCaseSource
from ripplecut.evidence.state_estimator import DOWN, DEGRADED, UP, StateEstimator
from ripplecut.model.loader import load_model

CASES_DIR = Path(__file__).resolve().parents[1] / "config" / "rcaeval_cases"


def test_parse_case_directory_name():
    benchmark, service, fault, instance = parse_case_directory_name("re1ob_cartservice_cpu_1")
    assert (benchmark, service, fault, instance) == ("re1ob", "cartservice", "cpu", 1)

    benchmark, service, fault, instance = parse_case_directory_name("re2ob_paymentservice_delay_3")
    assert (benchmark, service, fault, instance) == ("re2ob", "paymentservice", "delay", 3)


def test_rcaeval_adapter_cartservice_cpu(bundle):
    case_dir = CASES_DIR / "re1ob_cartservice_cpu_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    assert case_data.metadata.ground_truth_service == "cartservice"
    assert case_data.metadata.fault_type == "cpu"
    assert case_data.metadata.inject_time == 1692569340.0
    assert case_data.pre_window_count == 4
    assert case_data.post_window_count == 4

    cart_obs = case_data.normalized_observations["cartservice"]
    assert cart_obs.cpu_utilization >= 0.9
    assert cart_obs.error_rate >= 0.8
    assert cart_obs.latency_p95_ms >= 1400.0
    assert cart_obs.signals["cpu_utilization"].status is SignalStatus.AVAILABLE

    # Feed into StateEstimator
    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_observation_dict() for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate(bundle.system, obs_dict, source="RCAEval")

    descriptive = est_state.descriptive()
    assert descriptive["cartservice"] == DOWN

    # Evaluate RCA localization
    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_hit is True
    assert "cartservice" in rca_report.localized_services
    assert rca_report.ground_truth_service == "cartservice"


def test_rcaeval_adapter_paymentservice_delay(bundle):
    case_dir = CASES_DIR / "re1ob_paymentservice_delay_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    assert case_data.metadata.ground_truth_service == "paymentservice"
    assert case_data.metadata.fault_type == "delay"

    pay_obs = case_data.normalized_observations["paymentservice"]
    assert pay_obs.latency_p95_ms >= 2500.0
    assert pay_obs.error_rate >= 0.8

    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_observation_dict() for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate(bundle.system, obs_dict, source="RCAEval")

    descriptive = est_state.descriptive()
    assert descriptive["paymentservice"] == DOWN

    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_hit is True
    assert "paymentservice" in rca_report.localized_services


def test_rcaeval_adapter_missing_metrics_handling(bundle):
    """Phase 4.4: missing metrics handled explicitly via SignalStatus without silent failures."""
    case_dir = CASES_DIR / "re1ob_shippingservice_incomplete_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    ship_obs = case_data.normalized_observations["shippingservice"]
    assert ship_obs.signals["cpu_utilization"].status is SignalStatus.MISSING
    assert ship_obs.signals["error_rate"].status is SignalStatus.MISSING
    assert ship_obs.signals["latency_p95_ms"].status is SignalStatus.AVAILABLE

    # Missing metrics are recorded explicitly
    assert "cpu_utilization" in ship_obs.missing_signals
    assert "error_rate" in ship_obs.missing_signals

    # High latency classified as DEGRADED based on available latency signal
    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_observation_dict() for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate(bundle.system, obs_dict, source="RCAEval")

    descriptive = est_state.descriptive()
    assert descriptive["shippingservice"] == DEGRADED

    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_hit is True
    assert "cpu_utilization" in rca_report.signals_missing["shippingservice"]


def test_rcaeval_case_source_integration(bundle):
    case_dir = CASES_DIR / "re1ob_cartservice_cpu_1"
    source = RCAEvalCaseSource(case_dir, system=bundle.system, estimator_config=bundle.state_estimation)
    bundle_evidence = source.load()

    assert bundle_evidence.provenance["kind"] == "RCAEVAL_INSPECTED"
    assert bundle_evidence.provenance["case_id"] == "re1ob_cartservice_cpu_1"
    assert bundle_evidence.provenance["ground_truth_service"] == "cartservice"
    assert len(bundle_evidence.observations) == len(bundle.system.services)


def test_rcaeval_case_missing_directory(bundle):
    with pytest.raises(DataAdapterError, match="not found"):
        RCAEvalCaseSource(Path("nonexistent_case_dir"), system=bundle.system).load()


def test_rcaeval_metadata_and_anti_leakage(bundle, tmp_path):
    """Anti-leakage proof: ground_truth_service has zero impact on state estimation or normalized signals."""
    case_dir = CASES_DIR / "re1ob_cartservice_cpu_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data_real = adapter.load_case(case_dir)

    # Verify structured metadata fields
    assert case_data_real.metadata.source == "RCAEval"
    assert "github.com/phamquiluan/RCAEval" in case_data_real.metadata.source_reference
    assert case_data_real.metadata.retrieval_version == "1.0"
    assert case_data_real.metadata.ground_truth_service == "cartservice"

    estimator = StateEstimator(bundle.state_estimation)
    obs_real = {sid: o.to_observation_dict() for sid, o in case_data_real.normalized_observations.items()}
    state_real = estimator.estimate(bundle.system, obs_real, source="RCAEval")

    # Create a corrupted/altered case directory where ground_truth_service is completely changed or bogus
    corrupt_dir = tmp_path / "re1ob_bogusservice_fault_1"
    corrupt_dir.mkdir()
    (corrupt_dir / "inject_time.txt").write_text("1692569340\n", encoding="utf-8")
    (corrupt_dir / "metrics.json").write_text((case_dir / "metrics.json").read_text(encoding="utf-8"), encoding="utf-8")
    (corrupt_dir / "metadata.json").write_text(
        '{"source": "RCAEval", "case_id": "corrupt_1", "benchmark": "re1ob", '
        '"ground_truth_service": "totally_irrelevant_service", "fault_type": "cpu", '
        '"instance": 1, "inject_time": 1692569340.0, "source_reference": "ref", "retrieval_version": "1.0"}',
        encoding="utf-8"
    )

    case_data_corrupt = adapter.load_case(corrupt_dir)
    assert case_data_corrupt.metadata.ground_truth_service == "totally_irrelevant_service"

    # 1. Observations must be strictly identical across all services
    obs_corrupt = {sid: o.to_observation_dict() for sid, o in case_data_corrupt.normalized_observations.items()}
    assert obs_real == obs_corrupt

    for sid in bundle.system.service_ids:
        real_sig = case_data_real.normalized_observations[sid]
        corrupt_sig = case_data_corrupt.normalized_observations[sid]
        assert real_sig.latency_p95_ms == corrupt_sig.latency_p95_ms
        assert real_sig.error_rate == corrupt_sig.error_rate
        assert real_sig.cpu_utilization == corrupt_sig.cpu_utilization
        assert real_sig.request_rate == corrupt_sig.request_rate
        assert real_sig.missing_signals == corrupt_sig.missing_signals

    # 2. Estimated state must be strictly identical
    state_corrupt = estimator.estimate(bundle.system, obs_corrupt, source="RCAEval")
    assert state_real.formal_state(bundle.system) == state_corrupt.formal_state(bundle.system)
    assert state_real.descriptive() == state_corrupt.descriptive()

    # 3. Only evaluate_rca (reporting layer) uses ground_truth_service, and it correctly reports a miss
    report_corrupt = adapter.evaluate_rca(case_data_corrupt, state_corrupt.descriptive())
    assert report_corrupt.ground_truth_service == "totally_irrelevant_service"
    assert report_corrupt.root_cause_hit is False  # missed because ground truth was bogus

