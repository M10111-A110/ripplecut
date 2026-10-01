"""State estimation and evidence adapters (master spec §49-§52, §105)."""
import json

import pytest

from ripplecut.errors import DataAdapterError, ErrorCode
from ripplecut.evidence.sources import RCAEvalCaseSource, ReplayFixtureSource
from ripplecut.evidence.state_estimator import DEGRADED, DOWN, UP, Observation, StateEstimator
from ripplecut.model.loader import REPO_ROOT

FIXTURE = REPO_ROOT / "config" / "fixtures" / "synthetic_observed_checkout_down.json"


def obs(**kw):
    base = dict(service="s", latency_p95_ms=10, error_rate=0.0, request_rate=5, cpu_utilization=0.1)
    base.update(kw)
    return Observation(**base)


def test_thresholds_from_configuration(bundle):
    est = StateEstimator(bundle.state_estimation)
    assert est.classify(obs())[0] == UP
    assert est.classify(obs(error_rate=0.5))[0] == DOWN
    assert est.classify(obs(error_rate=0.05))[0] == DEGRADED
    assert est.classify(obs(latency_p95_ms=1000))[0] == DEGRADED
    assert est.classify(obs(cpu_utilization=0.95))[0] == DEGRADED
    custom = StateEstimator({**bundle.state_estimation, "down_error_rate_gte": 0.9})
    assert custom.classify(obs(error_rate=0.6))[0] == DEGRADED          # thresholds are parameters


def test_fixture_maps_to_expected_state(bundle):
    b = ReplayFixtureSource(FIXTURE).load()
    assert b.provenance["kind"] == "SYNTHETIC" and "SYNTHETIC" in b.label
    est = StateEstimator(bundle.state_estimation).estimate(bundle.system, b.observations, b.label)
    assert {k: v for k, v in est.descriptive().items() if v != UP} == {
        "paymentservice": DOWN, "checkoutservice": DOWN, "recommendationservice": DEGRADED}
    assert bundle.system.down_services(est.formal_state(bundle.system)) == (
        "checkoutservice", "paymentservice", "recommendationservice")        # DEGRADED -> 0


def test_missing_or_bad_observations_are_errors(bundle):
    est = StateEstimator(bundle.state_estimation)
    data = json.loads(FIXTURE.read_text())["observations"]
    missing = {k: v for k, v in data.items() if k != "cartservice"}
    with pytest.raises(DataAdapterError, match="missing observation"):
        est.estimate(bundle.system, missing, "t")
    with pytest.raises(DataAdapterError, match="outside the model"):
        est.estimate(bundle.system, {**data, "mysql": data["cartservice"]}, "t")
    for bad in ({"error_rate": 1.5}, {"latency_p95_ms": -1}, {"cpu_utilization": True}, {"request_rate": "x"}):
        with pytest.raises(DataAdapterError):
            est.estimate(bundle.system, {**data, "cartservice": {**data["cartservice"], **bad}}, "t")
    with pytest.raises(DataAdapterError):
        StateEstimator({"down_error_rate_gte": 0.5})
    with pytest.raises(DataAdapterError):
        StateEstimator({**bundle.state_estimation, "missing_observation": "ASSUME_UP"})


def test_fixture_requires_provenance(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"observations": {}}))
    with pytest.raises(DataAdapterError, match="provenance"):
        ReplayFixtureSource(p).load()
    p.write_text(json.dumps({"provenance": {"kind": "RCAEVAL_INSPECTED"}, "observations": {}}))
    with pytest.raises(DataAdapterError, match="case_id"):
        ReplayFixtureSource(p).load()


def test_rcaeval_adapter_refuses_to_guess(tmp_path):
    with pytest.raises(DataAdapterError) as exc:
        RCAEvalCaseSource(tmp_path / "RE2-OB_checkoutservice_cpu_1").load()
    assert exc.value.code is ErrorCode.DATA_ADAPTER_ERROR
    case = tmp_path / "some_case"
    case.mkdir()
    (case / "metrics.json").write_text("{}")
    with pytest.raises(DataAdapterError, match="REQUIRES VERIFICATION"):
        RCAEvalCaseSource(case).load()
