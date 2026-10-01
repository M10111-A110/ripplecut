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
