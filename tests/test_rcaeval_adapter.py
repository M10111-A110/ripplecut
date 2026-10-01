"""Tests for RCAEval adapter, official cases, and evaluation boundary (Phase 4, master spec §4.1-§4.5)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ripplecut.errors import DataAdapterError, ErrorCode
from ripplecut.evidence.rcaeval_adapter import (
    RCAEvalAdapter,
    SignalStatus,
    parse_case_directory_name,
    validate_directory_against_metadata,
)
from ripplecut.evidence.sources import RCAEvalCaseSource
from ripplecut.evidence.state_estimator import DOWN, DEGRADED, UP, EvidenceClassification, StateEstimator
from ripplecut.model.loader import load_model

CASES_DIR = Path(__file__).resolve().parents[1] / "config" / "rcaeval_cases"


def test_parse_case_directory_name():
    benchmark, service, fault, instance = parse_case_directory_name("re1ob_cartservice_cpu_1")
    assert (benchmark, service, fault, instance) == ("re1ob", "cartservice", "cpu", 1)

    benchmark, service, fault, instance = parse_case_directory_name("re2ob_paymentservice_delay_3")
    assert (benchmark, service, fault, instance) == ("re2ob", "paymentservice", "delay", 3)


def test_directory_name_disagreement_with_metadata_rejected(tmp_path, bundle):
    """Section 7: Directory name says cartservice but authoritative metadata says paymentservice -> DATA_ADAPTER_ERROR."""
    mismatch_dir = tmp_path / "re1ob_cartservice_cpu_1"
    mismatch_dir.mkdir()
    (mismatch_dir / "inject_time.txt").write_text("1692569340\n", encoding="utf-8")
    (mismatch_dir / "metrics.json").write_text('{"time": [1692569100, 1692569400]}', encoding="utf-8")
    (mismatch_dir / "metadata.json").write_text(
        json.dumps({
            "source": "SYNTHETIC_DERIVED",
            "case_id": "mismatch_1",
            "benchmark": "re1ob",
            "ground_truth_service": "paymentservice",  # Directory says cartservice!
            "fault_type": "cpu",
            "instance": 1,
            "inject_time": 1692569340.0,
        }),
        encoding="utf-8"
    )

    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    with pytest.raises(DataAdapterError, match="disagrees with authoritative metadata.*cartservice.*paymentservice"):
        adapter.load_case(mismatch_dir)

    # Fault type mismatch
    mismatch_fault_dir = tmp_path / "re1ob_cartservice_delay_1"
    mismatch_fault_dir.mkdir()
    (mismatch_fault_dir / "inject_time.txt").write_text("1692569340\n", encoding="utf-8")
    (mismatch_fault_dir / "metrics.json").write_text('{"time": [1692569100, 1692569400]}', encoding="utf-8")
    (mismatch_fault_dir / "metadata.json").write_text(
        json.dumps({
            "source": "SYNTHETIC_DERIVED",
            "case_id": "mismatch_2",
            "benchmark": "re1ob",
            "ground_truth_service": "cartservice",
            "fault_type": "cpu",  # Directory says delay!
            "instance": 1,
            "inject_time": 1692569340.0,
        }),
        encoding="utf-8"
    )
    with pytest.raises(DataAdapterError, match="disagrees with authoritative metadata.*delay.*cpu"):
        adapter.load_case(mismatch_fault_dir)


def test_inject_time_outside_range_fails_explicitly(tmp_path, bundle):
    """Section 8: Injection time outside timestamp range must fail explicitly, no 50/50 fallback."""
    bad_time_dir = tmp_path / "re1ob_cartservice_cpu_1"
    bad_time_dir.mkdir()
    (bad_time_dir / "inject_time.txt").write_text("1700000000\n", encoding="utf-8")  # Way in the future
    (bad_time_dir / "metrics.json").write_text(
        json.dumps({"time": [1692569100, 1692569200, 1692569300]}),
        encoding="utf-8"
    )
    (bad_time_dir / "metadata.json").write_text(
        json.dumps({
            "source": "SYNTHETIC_DERIVED",
            "case_id": "bad_time_1",
            "benchmark": "re1ob",
            "ground_truth_service": "cartservice",
            "fault_type": "cpu",
            "instance": 1,
            "inject_time": 1700000000.0,  # All timestamps are before inject_time!
        }),
        encoding="utf-8"
    )
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    with pytest.raises(DataAdapterError, match="inject_time.*does not meaningfully intersect"):
        adapter.load_case(bad_time_dir)


def test_rcaeval_adapter_official_cartservice_cpu(bundle):
    """Test official RCAEval case: re1ob_cartservice_cpu_1."""
    case_dir = CASES_DIR / "re1ob_cartservice_cpu_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    assert case_data.metadata.ground_truth_service == "cartservice"
    assert case_data.metadata.fault_type == "cpu"
    assert case_data.metadata.inject_time == 1685251430.0
    assert case_data.pre_window_count > 0
    assert case_data.post_window_count > 0

    cart_obs = case_data.normalized_observations["cartservice"]
    assert cart_obs.cpu_utilization is not None and cart_obs.cpu_utilization >= 0.9
    assert cart_obs.signals["cpu_utilization"].status is SignalStatus.AVAILABLE

    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_partial_observation_dict() for sid, o in case_data.normalized_observations.items()}
    sig_avail = {sid: {k: s.status.value for k, s in o.signals.items()} for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate_partial(bundle.system, obs_dict, source="RCAEval", signal_availability=sig_avail)

    descriptive = est_state.descriptive()
    assert descriptive["cartservice"] in (DOWN, DEGRADED)

    # Evaluate RCA localization
    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_detected is True
    assert "cartservice" in rca_report.localized_services
    assert rca_report.ground_truth_service == "cartservice"


def test_rcaeval_adapter_official_checkoutservice_delay(bundle):
    """Test official RCAEval case: re1ob_checkoutservice_delay_1."""
    case_dir = CASES_DIR / "re1ob_checkoutservice_delay_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    assert case_data.metadata.ground_truth_service == "checkoutservice"
    assert case_data.metadata.fault_type == "delay"
    assert case_data.metadata.inject_time == 1692602972.0

    checkout_obs = case_data.normalized_observations["checkoutservice"]
    assert checkout_obs.latency_p95_ms is not None and checkout_obs.latency_p95_ms >= 1000.0

    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_partial_observation_dict() for sid, o in case_data.normalized_observations.items()}
    sig_avail = {sid: {k: s.status.value for k, s in o.signals.items()} for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate_partial(bundle.system, obs_dict, source="RCAEval", signal_availability=sig_avail)

    descriptive = est_state.descriptive()
    assert descriptive["checkoutservice"] in (DOWN, DEGRADED)

    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_detected is True
    assert "checkoutservice" in rca_report.localized_services


def test_rcaeval_adapter_official_productcatalogservice_cpu(bundle):
    """Test official RCAEval case: re1ob_productcatalogservice_cpu_1."""
    case_dir = CASES_DIR / "re1ob_productcatalogservice_cpu_1"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(case_dir)

    assert case_data.metadata.ground_truth_service == "productcatalogservice"
    assert case_data.metadata.fault_type == "cpu"
    assert case_data.metadata.inject_time == 1685364577.0

    catalog_obs = case_data.normalized_observations["productcatalogservice"]
    assert catalog_obs.cpu_utilization is not None and catalog_obs.cpu_utilization >= 0.9

    estimator = StateEstimator(bundle.state_estimation)
    obs_dict = {sid: o.to_partial_observation_dict() for sid, o in case_data.normalized_observations.items()}
    sig_avail = {sid: {k: s.status.value for k, s in o.signals.items()} for sid, o in case_data.normalized_observations.items()}
    est_state = estimator.estimate_partial(bundle.system, obs_dict, source="RCAEval", signal_availability=sig_avail)

    descriptive = est_state.descriptive()
    assert descriptive["productcatalogservice"] in (DOWN, DEGRADED)

    rca_report = adapter.evaluate_rca(case_data, descriptive)
    assert rca_report.root_cause_detected is True
    assert "productcatalogservice" in rca_report.localized_services


def test_rcaeval_case_source_integration(bundle):
    case_dir = CASES_DIR / "re1ob_cartservice_cpu_1"
    source = RCAEvalCaseSource(case_dir, system=bundle.system, estimator_config=bundle.state_estimation)
    bundle_evidence = source.load()

    assert bundle_evidence.provenance["kind"] in ("OFFICIAL_RCAEVAL", "RCAEVAL_INSPECTED")
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
    assert case_data_real.metadata.ground_truth_service == "cartservice"

    estimator = StateEstimator(bundle.state_estimation)
    obs_real = {sid: o.to_partial_observation_dict() for sid, o in case_data_real.normalized_observations.items()}
    sig_avail_real = {sid: {k: s.status.value for k, s in o.signals.items()} for sid, o in case_data_real.normalized_observations.items()}
    state_real = estimator.estimate_partial(bundle.system, obs_real, source="RCAEval", signal_availability=sig_avail_real)

    # Create an altered case directory matching directory name with metadata
    corrupt_dir = tmp_path / "re1ob_totallyirrelevant_cpu_1"
    corrupt_dir.mkdir()
    (corrupt_dir / "inject_time.txt").write_text(f"{int(case_data_real.metadata.inject_time)}\n", encoding="utf-8")
    (corrupt_dir / "metrics.json").write_text((case_dir / "metrics.json").read_text(encoding="utf-8"), encoding="utf-8")
    if (case_dir / "metrics.parquet").exists():
        (corrupt_dir / "metrics.parquet").write_bytes((case_dir / "metrics.parquet").read_bytes())
    (corrupt_dir / "metadata.json").write_text(
        json.dumps({
            "source": "RCAEval",
            "case_id": "corrupt_1",
            "benchmark": "re1ob",
            "ground_truth_service": "totallyirrelevant",
            "fault_type": "cpu",
            "instance": 1,
            "inject_time": case_data_real.metadata.inject_time,
            "source_reference": "ref",
            "retrieval_version": "1.0"
        }),
        encoding="utf-8"
    )

    case_data_corrupt = adapter.load_case(corrupt_dir)
    assert case_data_corrupt.metadata.ground_truth_service == "totallyirrelevant"

    # 1. Observations must be strictly identical across all services
    obs_corrupt = {sid: o.to_partial_observation_dict() for sid, o in case_data_corrupt.normalized_observations.items()}
    sig_avail_corrupt = {sid: {k: s.status.value for k, s in o.signals.items()} for sid, o in case_data_corrupt.normalized_observations.items()}

    for sid in bundle.system.service_ids:
        real_sig = case_data_real.normalized_observations[sid]
        corrupt_sig = case_data_corrupt.normalized_observations[sid]
        assert real_sig.latency_p95_ms == corrupt_sig.latency_p95_ms
        assert real_sig.error_rate == corrupt_sig.error_rate
        assert real_sig.cpu_utilization == corrupt_sig.cpu_utilization
        assert real_sig.request_rate == corrupt_sig.request_rate
        assert real_sig.missing_signals == corrupt_sig.missing_signals

    # 2. Estimated state must be strictly identical
    state_corrupt = estimator.estimate_partial(bundle.system, obs_corrupt, source="RCAEval", signal_availability=sig_avail_corrupt)
    assert state_real.formal_state(bundle.system) == state_corrupt.formal_state(bundle.system)
    assert state_real.descriptive() == state_corrupt.descriptive()

    # 3. Only evaluate_rca (reporting layer) uses ground_truth_service, and it correctly reports a miss
    report_corrupt = adapter.evaluate_rca(case_data_corrupt, state_corrupt.descriptive())
    assert report_corrupt.ground_truth_service == "totallyirrelevant"
    assert report_corrupt.root_cause_detected is False  # missed because ground truth was bogus
