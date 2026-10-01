"""Comprehensive test matrix for missing-signal semantics and partial evidence (Section 12).

Tests:
- Case A: All signals available and healthy -> UP (SUFFICIENT)
- Case B: High error rate -> DOWN (SUFFICIENT or PARTIAL)
- Case C: High latency but no error -> DEGRADED (SUFFICIENT or PARTIAL)
- Case D: CPU missing, other evidence healthy -> UP (PARTIAL, with explicit missing_signals)
- Case E: Error rate missing, other evidence healthy -> UP (PARTIAL, explicitly NOT assuming zero errors)
- Case F: All signals missing -> INSUFFICIENT_EVIDENCE (formal=0)
- Case G: High CPU, error rate missing -> DEGRADED (PARTIAL)
- Case H: Error rate missing, latency high -> DEGRADED (PARTIAL)
- Case I: Strict mode rejects missing signals with DataAdapterError
- Case J: Multiple services with mixed evidence completeness
- Case K: Missing signals are not converted to 0.0 or healthy defaults
- Case L: Formal Boolean state preserves conservative safety under partial evidence
"""
from __future__ import annotations

import pytest

from ripplecut.errors import DataAdapterError
from ripplecut.evidence.rcaeval_adapter import SignalStatus
from ripplecut.evidence.state_estimator import (
    DEGRADED,
    DOWN,
    INSUFFICIENT_EVIDENCE,
    UP,
    EvidenceClassification,
    Observation,
    StateEstimator,
    parse_observation,
    parse_partial_observation,
)


@pytest.fixture
def estimator(bundle):
    return StateEstimator(bundle.state_estimation)


def test_case_a_all_signals_available_and_healthy(bundle, estimator):
    """Case A: All signals available and healthy -> UP with SUFFICIENT evidence."""
    obs = {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
    state, reason, evidence = estimator.classify_partial(
        parse_partial_observation("cartservice", obs, ()),
        ()
    )
    assert state == UP
    assert evidence == EvidenceClassification.SUFFICIENT.value
    assert "below configured thresholds" in reason


def test_case_b_high_error_rate(bundle, estimator):
    """Case B: High error rate -> DOWN (error_rate >= 0.5)."""
    obs = {"latency_p95_ms": 10.0, "error_rate": 0.8, "request_rate": 50.0, "cpu_utilization": 0.1}
    state, reason, evidence = estimator.classify_partial(
        parse_partial_observation("cartservice", obs, ()),
        ()
    )
    assert state == DOWN
    assert evidence == EvidenceClassification.SUFFICIENT.value
    assert "error_rate" in reason


def test_case_c_high_latency_no_error(bundle, estimator):
    """Case C: High latency (>=1000ms) but zero error rate -> DEGRADED."""
    obs = {"latency_p95_ms": 1500.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
    state, reason, evidence = estimator.classify_partial(
        parse_partial_observation("cartservice", obs, ()),
        ()
    )
    assert state == DEGRADED
    assert evidence == EvidenceClassification.SUFFICIENT.value
    assert "latency_p95_ms" in reason


def test_case_d_cpu_missing_other_healthy(bundle, estimator):
    """Case D: CPU missing, other evidence healthy -> UP with PARTIAL evidence and cpu_utilization tracked."""
    obs = {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0}
    missing = ("cpu_utilization",)
    parsed = parse_partial_observation("cartservice", obs, missing)
    assert parsed.cpu_utilization is None  # Genuinely None, not 0.0!

    state, reason, evidence = estimator.classify_partial(parsed, missing)
    assert state == UP
    assert evidence == EvidenceClassification.PARTIAL.value
    assert "missing: cpu_utilization" in reason


def test_case_e_error_rate_missing_other_healthy(bundle, estimator):
    """Case E: Error rate missing, other evidence healthy -> UP with PARTIAL evidence, NOT silently assuming 0 errors."""
    obs = {"latency_p95_ms": 10.0, "request_rate": 50.0, "cpu_utilization": 0.1}
    missing = ("error_rate",)
    parsed = parse_partial_observation("cartservice", obs, missing)
    assert parsed.error_rate is None  # Genuinely None, not 0.0!

    state, reason, evidence = estimator.classify_partial(parsed, missing)
    assert state == UP
    assert evidence == EvidenceClassification.PARTIAL.value
    assert "missing: error_rate" in reason
    # Must NOT say "all signals below configured thresholds" (which implies error_rate was verified)
    assert "available signals below thresholds" in reason


def test_case_f_all_signals_missing(bundle, estimator):
    """Case F: All signals missing -> INSUFFICIENT evidence, formal state 0."""
    obs = {}
    missing = ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")
    parsed = parse_partial_observation("cartservice", obs, missing)
    assert all(getattr(parsed, f) is None for f in missing)

    state, reason, evidence = estimator.classify_partial(parsed, missing)
    assert evidence == EvidenceClassification.INSUFFICIENT.value
    assert "no signals available" in reason


def test_case_g_high_cpu_error_rate_missing(bundle, estimator):
    """Case G: High CPU (>=0.9) with error_rate missing -> DEGRADED with PARTIAL evidence."""
    obs = {"latency_p95_ms": 10.0, "request_rate": 50.0, "cpu_utilization": 0.95}
    missing = ("error_rate",)
    parsed = parse_partial_observation("cartservice", obs, missing)

    state, reason, evidence = estimator.classify_partial(parsed, missing)
    assert state == DEGRADED
    assert evidence == EvidenceClassification.PARTIAL.value
    assert "cpu_utilization" in reason


def test_case_h_high_latency_error_rate_missing(bundle, estimator):
    """Case H: High latency (>=1000ms) with error_rate missing -> DEGRADED with PARTIAL evidence."""
    obs = {"latency_p95_ms": 2500.0, "request_rate": 50.0, "cpu_utilization": 0.2}
    missing = ("error_rate",)
    parsed = parse_partial_observation("cartservice", obs, missing)

    state, reason, evidence = estimator.classify_partial(parsed, missing)
    assert state == DEGRADED
    assert evidence == EvidenceClassification.PARTIAL.value
    assert "latency_p95_ms" in reason


def test_case_i_strict_mode_rejects_missing_signals(bundle, estimator):
    """Case I: In strict mode (standard estimate()), missing signals raise DataAdapterError."""
    valid_obs = {
        sid: {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
        for sid in bundle.system.service_ids
    }
    # Delete error_rate from cartservice
    del valid_obs["cartservice"]["error_rate"]

    with pytest.raises(DataAdapterError, match="missing \\['error_rate'\\]"):
        estimator.estimate(bundle.system, valid_obs, "test_strict")


def test_case_j_mixed_evidence_across_services(bundle, estimator):
    """Case J: Multiple services with mixed evidence completeness."""
    obs_dict = {}
    sig_avail = {}

    for sid in bundle.system.service_ids:
        if sid == "cartservice":
            # Fully available, healthy
            obs_dict[sid] = {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
            sig_avail[sid] = {f: "AVAILABLE" for f in ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")}
        elif sid == "paymentservice":
            # Missing cpu and error_rate, high latency
            obs_dict[sid] = {"latency_p95_ms": 2500.0, "request_rate": 10.0}
            sig_avail[sid] = {
                "latency_p95_ms": "AVAILABLE",
                "request_rate": "AVAILABLE",
                "cpu_utilization": "MISSING",
                "error_rate": "MISSING",
            }
        elif sid == "shippingservice":
            # All missing!
            obs_dict[sid] = {}
            sig_avail[sid] = {f: "MISSING" for f in ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")}
        else:
            obs_dict[sid] = {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
            sig_avail[sid] = {f: "AVAILABLE" for f in ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")}

    est = estimator.estimate_partial(bundle.system, obs_dict, "test_mixed", sig_avail)

    cart = next(s for s in est.services if s.service_id == "cartservice")
    assert cart.state == UP
    assert cart.formal == 1
    assert cart.evidence == EvidenceClassification.SUFFICIENT.value

    pay = next(s for s in est.services if s.service_id == "paymentservice")
    assert pay.state == DEGRADED
    assert pay.formal == 0
    assert pay.evidence == EvidenceClassification.PARTIAL.value

    ship = next(s for s in est.services if s.service_id == "shippingservice")
    assert ship.state == INSUFFICIENT_EVIDENCE
    assert ship.formal == 0
    assert ship.evidence == EvidenceClassification.INSUFFICIENT.value


def test_case_k_missing_signals_never_default_to_zero(bundle):
    """Case K: Verify that missing signals in NormalizedServiceObservation are None, not 0.0."""
    from ripplecut.evidence.rcaeval_adapter import RCAEvalAdapter
    from pathlib import Path
    cases_dir = Path(__file__).resolve().parents[1] / "config" / "rcaeval_cases"
    adapter = RCAEvalAdapter(bundle.system, bundle.state_estimation)
    case_data = adapter.load_case(cases_dir / "re1ob_shippingservice_incomplete_1")

    ship_obs = case_data.normalized_observations["shippingservice"]
    # The crucial invariant:
    assert ship_obs.signals["cpu_utilization"].value is None
    assert ship_obs.signals["error_rate"].value is None
    assert ship_obs.cpu_utilization is None
    assert ship_obs.error_rate is None


def test_case_l_formal_state_conservative_safety(bundle, estimator):
    """Case L: Formal binary state x preserves conservative safety (non-UP maps to 0)."""
    obs_dict = {
        sid: {"latency_p95_ms": 10.0, "error_rate": 0.0, "request_rate": 50.0, "cpu_utilization": 0.1}
        for sid in bundle.system.service_ids
    }
    # shippingservice has all signals missing
    obs_dict["shippingservice"] = {}
    sig_avail = {
        sid: {f: "AVAILABLE" for f in ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")}
        for sid in bundle.system.service_ids
    }
    sig_avail["shippingservice"] = {f: "MISSING" for f in ("latency_p95_ms", "error_rate", "request_rate", "cpu_utilization")}

    est = estimator.estimate_partial(bundle.system, obs_dict, "test_safety", sig_avail)
    formal = est.formal_state(bundle.system)
    ship_idx = bundle.system.index("shippingservice")
    # INSUFFICIENT_EVIDENCE must NOT be treated as UP (formal=0)
    assert formal[ship_idx] == 0
