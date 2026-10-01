"""Reproducibility (master spec §46, §63) and structured logging (§66)."""
from ripplecut.generators import random_problem
from ripplecut.pipeline import fingerprint
from ripplecut.solvers.guard import ExecutionGuard
from ripplecut.solvers.branch_and_bound import BranchAndBoundSolver


def test_same_input_same_output(app):
    for sid in [s["id"] for s in app.scenarios]:
        assert fingerprint(app.run_scenario(sid)) == fingerprint(app.run_scenario(sid))


def test_input_order_is_irrelevant(app):
    a = app.run_incident({"failed_services": ["shippingservice", "paymentservice"]})
    b = app.run_incident({"failed_services": ["paymentservice", "shippingservice"]})
    assert a["input"] != b["input"]                           # the raw input is echoed verbatim ...
    strip = lambda r: {k: v for k, v in r.items() if k != "input"}
    assert fingerprint(strip(a)) == fingerprint(strip(b))     # ... everything derived from it is identical


def test_seeded_generation_is_deterministic():
    p1, _ = random_problem(123, n_services=9, m_actions=7)
    p2, _ = random_problem(123, n_services=9, m_actions=7)
    assert p1.initial_state == p2.initial_state and p1.metadata["seed"] == 123
    r1 = ExecutionGuard().run(BranchAndBoundSolver(), p1)
    r2 = ExecutionGuard().run(BranchAndBoundSolver(), p2)
    assert (r1.status, r1.selected_plan, r1.nodes_explored) == (r2.status, r2.selected_plan, r2.nodes_explored)


def test_structured_log_events(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value")
    r = app.run_scenario("payment_failure")
    kinds = [e["event"] for e in r["events"]]
    for needed in ("initial_state", "cascade_round", "solver_attempt", "validation", "final_plan", "objective_values"):
        assert needed in kinds, needed
    attempt = next(e for e in r["events"] if e["event"] == "solver_attempt")
    assert {"solver", "status", "runtime_seconds"} <= set(attempt)
    assert "sk-test-secret-value" not in str(r)
    fb = app.run_scenario("payment_failure", fault=__import__("ripplecut.pipeline", fromlist=["FaultSpec"]).FaultSpec("crash"))
    assert "fallback" in [e["event"] for e in fb["events"]]
