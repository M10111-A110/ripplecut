"""GATES 8-12 - canonical scenarios, end-to-end flow, fallback under injected faults, offline demo, UI API."""
import json
import socket

import pytest

from ripplecut import cli
from ripplecut.cli import check_expectation
from ripplecut.pipeline import FaultSpec, deterministic_view
from ripplecut.solvers.fault_injection import FAULT_MODES
from ripplecut.ui.server import DemoState, handle_api
from ripplecut.errors import RippleCutError

SCENARIOS = json.loads((cli.Path(__file__).resolve().parents[1] / "config" / "scenarios.json").read_text())["scenarios"]
IDS = [s["id"] for s in SCENARIOS]


@pytest.mark.parametrize("sid", IDS)
def test_canonical_scenario_matches_hand_derived_expectation(app, sid):
    """GATE 10."""
    scn = app.scenario(sid)
    r = app.run_scenario(sid)
    assert check_expectation(scn, r) == [], (sid, check_expectation(scn, r))
    assert r["label"].startswith("Controlled RippleCut Scenario")
    if r["status"] == "SUCCESS":
        assert r["result"]["validation_status"] == "VALID"
        assert r["result"]["optimality_status"] == "PROVEN_OPTIMAL"
        assert r["planner"]["verification"]["status"] == "EQUIVALENT"
        assert r["approval"]["status"] == "PENDING_HUMAN_APPROVAL"


def test_at_least_four_scenarios_one_topology(app):
    assert len(IDS) >= 4
    models = {app.run_scenario(sid)["model"]["id"] for sid in IDS}
    assert models == {"online_boutique_canonical_v1"}
    plans = {json.dumps(app.run_scenario(sid)["result"].get("plan")) for sid in IDS}
    assert len(plans) >= 5                                             # different incidents, different plans


@pytest.mark.parametrize("sid", [s["id"] for s in SCENARIOS if s.get("example_text")])
def test_end_to_end_from_natural_language(app, sid):
    """GATE 11 / §92: text -> structured incident -> state -> cascade -> solver -> validator -> explanation."""
    via_text = app.run_scenario(sid, via_text=True)
    structured = app.run_scenario(sid)
    assert via_text["parse"]["status"] == "OK"
    assert via_text["incident"]["failed_services"] == structured["incident"]["failed_services"]
    for key in ("status", "uncontrolled_cascade", "result", "contained_cascade"):
        assert deterministic_view(via_text.get(key)) == deterministic_view(structured.get(key)), key
    assert via_text["explanation"]["text"] == structured["explanation"]["text"]


@pytest.mark.parametrize("mode", FAULT_MODES)
@pytest.mark.parametrize("sid", ["payment_failure", "payment_and_shipping_failure", "cart_failure",
                                 "observed_state_replay"])
def test_resilience_every_fault_mode(app, sid, mode):
    """GATES 8/9: the primary solver is broken; the fallback result is validated and identical."""
    normal = app.run_scenario(sid)
    broken = app.run_scenario(sid, fault=FaultSpec(mode, timeout_seconds=0.3))
    assert broken["resilience_test"]["label"] == "Solver failure simulation / resilience test"
    first = broken["planner"]["attempts"][0]
    assert first["solver"] == broken["resilience_test"]["target_solver"]
    if mode == "wrong_cost" and normal["status"] == "NO_FEASIBLE_PLAN":
        # an infeasibility result has no cost to corrupt: the (correct, certified) result is passed through
        assert first["outcome"] == "ACCEPTED" and "not applicable" in first["result"]["metadata"]["fault_injection"]
    else:
        expected_first = {"crash": "ERROR", "timeout": "TIMEOUT", "malformed": "INVALID_RESULT",
                          "invalid_action": "INVALID_RESULT", "wrong_cost": "INVALID_RESULT"}[mode]
        assert first["outcome"] in ("FAILED", "REJECTED_BY_VALIDATOR")
        assert first["result"]["status"] == expected_first
        assert broken["result"]["fallback_count"] == 1
    assert broken["status"] == normal["status"]
    assert broken["result"].get("plan") == normal["result"].get("plan")


def test_explanation_is_derived_from_results(app):
    r = app.run_scenario("payment_and_shipping_failure")
    text = r["explanation"]["text"]
    assert "checkout_backend_standby" in text and "same K, but higher number of interventions N (2 vs 1)" in text
    assert "PROVEN_OPTIMAL" in text and "human must approve" in text
    inf = app.run_scenario("cart_failure")["explanation"]["text"]
    assert "cannot preserve all designated critical services" in inf and "cartservice" in inf


def test_input_errors_become_statuses(app):
    assert app.run_incident({"failed_services": ["ghost"]})["status"] == "INVALID_SERVICE"
    assert app.run_text("the database is down")["status"] == "INVALID_SERVICE"
    amb = app.run_text("Payment service is down and checkout orders are beginning to fail.")
    assert amb["status"] == "INCIDENT_AMBIGUOUS" and "uncontrolled_cascade" not in amb
    assert app.run_incident({"failed_services": []})["result"]["plan"] == []          # no failures: do nothing


def test_demo_runs_offline(monkeypatch, capsys):
    """GATE 12: the full demo never touches the network."""
    def no_network(*a, **k):
        raise AssertionError("network access attempted during the offline demo")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    assert cli.main(["demo", "--fault-timeout", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "8/8 match" in out and "Solver failure simulation / resilience test" in out


def test_cli_commands(capsys):
    assert cli.main(["validate-config"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert cli.main(["run", "--failed", "paymentservice", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["plan"] == ["payment_fallback"]
    assert cli.main(["run", "--text", "authservice is down"]) == 2
    assert cli.main(["run", "--scenario", "cart_failure"]) == 0
    assert cli.main(["scenarios"]) == 0


def test_ui_api_contract(app, tmp_path):
    """The UI is a view over run reports; approval is simulated and executes nothing (§72)."""
    state = DemoState(app, audit_dir=tmp_path / "audit")
    model = handle_api(state, "GET", "/api/model", {})
    assert len(model["services"]) == 11 and model["rule_semantics_label"] == "RIPPLECUT-MODELED RULE"
    assert handle_api(state, "POST", "/api/parse", {"text": "paymentservice is down"})["status"] == "OK"
    r = handle_api(state, "POST", "/api/run", {"scenario_id": "payment_failure", "fault": {"mode": "crash"}})
    assert r["result"]["selected_solver"] != r["resilience_test"]["target_solver"]
    rec = handle_api(state, "POST", "/api/approve", {"run_id": r["run_id"]})
    assert rec["status"] == "APPROVED_SIMULATED" and rec["executed"] is False
    cart = handle_api(state, "POST", "/api/run", {"incident": {"failed_services": ["cartservice"]}})
    with pytest.raises(RippleCutError):
        handle_api(state, "POST", "/api/approve", {"run_id": cart["run_id"]})
    with pytest.raises(RippleCutError):
        handle_api(state, "POST", "/api/run", {})


def test_all_critical_services_already_down(app):
    """Master spec §64 case 4: every critical service DOWN in x(0) -> certified NO_FEASIBLE_PLAN, no crash."""
    crit = sorted(app.bundle.system.critical)
    r = app.run_incident({"failed_services": crit})
    assert r["status"] == "NO_FEASIBLE_PLAN" and r["result"]["infeasibility_certified"] is True
    assert r["uncontrolled_cascade"]["state_changing_rounds"] >= 0


def test_ui_audit_logging_and_fault_guard(app, tmp_path, monkeypatch):
    """Verify persistent audit logging on /api/approve and fault injection toggle."""
    monkeypatch.chdir(tmp_path)
    state = DemoState(app, allow_fault_injection=False)
    with pytest.raises(RippleCutError, match="fault injection is disabled"):
        handle_api(state, "POST", "/api/run", {"scenario_id": "payment_failure", "fault": {"mode": "crash"}})

    state.allow_fault_injection = True
    r = handle_api(state, "POST", "/api/run", {"scenario_id": "payment_failure"})
    rec = handle_api(state, "POST", "/api/approve", {"run_id": r["run_id"]})
    assert rec["status"] == "APPROVED_SIMULATED"
    audit_file = tmp_path / "audit" / "approvals.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["run_id"] == r["run_id"]
    assert logged["plan"] == ["payment_fallback"]

