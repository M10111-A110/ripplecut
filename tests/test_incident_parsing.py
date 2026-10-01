"""Incident input, deterministic parser and the optional LLM boundary (master spec §6.1, §7, §62 'LLM', §71)."""
import json

import pytest

from ripplecut.errors import ErrorCode, IncidentError
from ripplecut.explain.llm_explainer import check_explanation, llm_explanation
from ripplecut.incident.llm import LLMIncidentParser, LLMUnavailable
from ripplecut.incident.parser import AMBIGUOUS, LLM_PARSE_ERROR, NO_SERVICE_FOUND, OK, UNKNOWN_SERVICE, RuleBasedParser
from ripplecut.incident.schema import validate_incident


class StubLLM:
    name = "stub"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture(scope="module")
def parser(bundle):
    return RuleBasedParser(bundle.system, bundle.raw)


# ---- structured input ---------------------------------------------------------------------------------------------
def test_structured_incident_is_canonical(bundle):
    inc = validate_incident({"failed_services": ["shippingservice", "paymentservice", "paymentservice"],
                             "scenario_id": "x"}, bundle.system)
    assert inc.failed_services == ("paymentservice", "shippingservice")
    assert inc.canonical() == {"failed_services": ["paymentservice", "shippingservice"], "scenario_id": "x"}


@pytest.mark.parametrize("raw,code", [
    ({"failed_services": ["authservice"]}, ErrorCode.INVALID_SERVICE),
    ({"failed_services": ["paymentservice"], "actions": ["payment_fallback"]}, ErrorCode.CONFIG_ERROR),
    ({"failed_services": "paymentservice"}, ErrorCode.CONFIG_ERROR),
    ({"failed_services": ["paymentservice"], "degraded_services": ["paymentservice"]}, ErrorCode.INCIDENT_AMBIGUOUS),
    ({"failed_services": [], "scenario_id": "bad id!"}, ErrorCode.CONFIG_ERROR),
    ({}, ErrorCode.CONFIG_ERROR),
    (["paymentservice"], ErrorCode.CONFIG_ERROR),
])
def test_structured_incident_rejections(bundle, raw, code):
    with pytest.raises(IncidentError) as exc:
        validate_incident(raw, bundle.system)
    assert exc.value.code is code


def test_degraded_maps_to_zero(bundle):
    inc = validate_incident({"failed_services": [], "degraded_services": ["recommendationservice"]}, bundle.system)
    assert bundle.system.down_services(inc.initial_state(bundle.system)) == ("recommendationservice",)


# ---- deterministic parser -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["paymentservice is down", "Payment service is DOWN.", "payments are failing",
                                  "the payment service crashed", "Outage: payment", "payment is unavailable",
                                  "payment service not responding"])
def test_multiple_phrasings_same_incident(parser, text):
    r = parser.parse(text)
    assert r.status == OK and r.incident.failed_services == ("paymentservice",), (text, r.message)


def test_multiple_services_and_states(parser):
    r = parser.parse("payment and shipping are down; recommendations are slow; ads are fine")
    assert r.status == OK
    assert r.incident.failed_services == ("paymentservice", "shippingservice")
    assert r.incident.degraded_services == ("recommendationservice",)


@pytest.mark.parametrize("text,status", [
    ("the database is down", UNKNOWN_SERVICE),
    ("authservice is failing", UNKNOWN_SERVICE),
    ("redis is down", UNKNOWN_SERVICE),
    ("everything is broken", NO_SERVICE_FOUND),
    ("", NO_SERVICE_FOUND),
    ("checkout", AMBIGUOUS),
    ("payment is not down", AMBIGUOUS),
    ("payment is down and payment is fine", AMBIGUOUS),
])
def test_parser_refuses_to_guess(parser, text, status):
    r = parser.parse(text)
    assert r.status == status and r.incident is None


def test_symptom_or_root_ambiguity_offers_alternatives(parser):
    r = parser.parse("Payment service is down and checkout orders are beginning to fail.")
    assert r.status == AMBIGUOUS
    alts = {a["id"]: a["incident"]["failed_services"] for a in r.alternatives}
    assert alts == {"consequence": ["paymentservice"], "independent": ["checkoutservice", "paymentservice"]}


# ---- LLM boundary (stub client; no network) --------------------------------------------------------------------
def test_llm_valid_structured_output(bundle):
    stub = StubLLM(json.dumps({"failed_services": ["paymentservice"], "degraded_services": [], "unknown_mentions": [],
                               "ambiguous": False, "ambiguity_reason": ""}))
    r = LLMIncidentParser(bundle.system, stub).parse("Payments are broken")
    assert r.status == OK and r.incident.failed_services == ("paymentservice",) and r.incident.source == "llm_parser"
    assert "paymentservice" in stub.prompts[0][0]                    # allowed ids are given, config is not


def test_llm_fenced_json_and_phrasings(bundle):
    reply = "```json\n" + json.dumps({"failed_services": ["shippingservice"]}) + "\n```"
    for text in ("shipping down", "Shipping service outage", "cannot ship orders"):
        r = LLMIncidentParser(bundle.system, StubLLM(reply)).parse(text)
        assert r.status == OK and r.incident.failed_services == ("shippingservice",)


@pytest.mark.parametrize("reply,status", [
    ("the payment service is down", LLM_PARSE_ERROR),                                     # malformed (not JSON)
    ("[1, 2]", LLM_PARSE_ERROR),
    (json.dumps({"failed_services": ["paymentservice"], "recommended_actions": ["reboot_everything"]}),
     LLM_PARSE_ERROR),                                                                    # hallucinated action
    (json.dumps({"failed_services": ["paymentservice"], "cost": 0}), LLM_PARSE_ERROR),
    (json.dumps({"failed_services": ["authservice"]}), UNKNOWN_SERVICE),                  # unknown service
    (json.dumps({"failed_services": [], "unknown_mentions": ["mysql"]}), UNKNOWN_SERVICE),
    (json.dumps({"failed_services": ["cartservice"], "ambiguous": True, "ambiguity_reason": "cart or checkout?"}),
     AMBIGUOUS),                                                                          # ambiguous service
    (json.dumps({"failed_services": "cartservice"}), LLM_PARSE_ERROR),
    (json.dumps({"failed_services": ["paymentservice", "checkoutservice"]}), AMBIGUOUS),  # root vs symptom
])
def test_llm_output_is_schema_validated(bundle, reply, status):
    r = LLMIncidentParser(bundle.system, StubLLM(reply)).parse("some incident")
    assert r.status == status and r.incident is None


def test_llm_unavailable_falls_back_to_deterministic_parser(bundle):
    from ripplecut.pipeline import RippleCutApp
    app = RippleCutApp(llm_client=StubLLM(LLMUnavailable("network down")))
    r = app.parse_text("paymentservice is down", use_llm=True)
    assert r.status == OK and r.parser == "rule_based_parser" and any("unavailable" in n for n in r.notes)
    offline = RippleCutApp().parse_text("paymentservice is down", use_llm=True)
    assert offline.status == OK and any("offline" in n for n in offline.notes)


# ---- LLM explanation guard (§71) ---------------------------------------------------------------------------------
FACTS = {"initial_failures": ["paymentservice"], "selected_plan": [{"id": "payment_fallback", "cost": 3}],
         "objective": {"K": 3, "N": 1, "R": 0}, "runner_ups": [{"plan": ["checkout_backend_standby"], "K": 5}]}


def test_llm_explanation_guard_accepts_faithful_text(bundle):
    text = ("paymentservice failed, so checkoutservice lost a hard dependency. The recommended plan is "
            "payment_fallback at a cost of 3; checkout_backend_standby would cost 5. A human must approve it.")
    assert check_explanation(text, FACTS, bundle.system, bundle.actions) == []


@pytest.mark.parametrize("text,needle", [
    ("Restart the authservice to fix it.", "unknown service"),
    ("Use reboot_cluster immediately.", "unknown action"),
    ("Use restore_currency_replica as well.", "not part of the provided facts"),
    ("payment_fallback has a cost of 1.", "cost 1"),
    ("RippleCut automatically fixed the outage with payment_fallback.", "forbidden claim"),
])
def test_llm_explanation_guard_rejects_inventions(bundle, text, needle):
    issues = check_explanation(text, FACTS, bundle.system, bundle.actions)
    assert any(needle in i for i in issues), issues


def test_llm_explanation_falls_back_on_error(bundle):
    out = llm_explanation(StubLLM(RuntimeError("down")), FACTS, bundle.system, bundle.actions)
    assert not out["accepted"] and out["text"] is None


# ---- temporal qualifiers, uncertainty, recovery, and retraction (§104-§105) --------------------------------------
def test_past_failure_with_recovery_resolves_healthy(parser):
    r = parser.parse("payment failed five minutes ago but is healthy now")
    assert r.status == OK
    assert r.incident is not None
    assert r.incident.failed_services == ()
    assert r.incident.degraded_services == ()
    assert any("paymentservice" in n for n in r.notes)


def test_past_failure_with_recovery_and_active_failure(parser):
    r = parser.parse("payment failed five minutes ago but is healthy now; cartservice is down")
    assert r.status == OK
    assert r.incident is not None
    assert r.incident.failed_services == ("cartservice",)
    assert any("paymentservice" in n for n in r.notes)


@pytest.mark.parametrize("text,expected_phrase", [
    ("payment was down earlier", "historical report without current confirmation"),
    ("payment had failed 10 minutes ago", "historical report without current confirmation"),
    ("payment might be down", "uncertain state"),
    ("could be that paymentservice is degraded", "uncertain state"),
    ("payment is down. just kidding, it's up", "retracted statement"),
    ("payment is failing - never mind, false alarm", "retracted statement"),
    ("what if payment fails", "hypothetical scenario"),
    ("suppose paymentservice is down", "hypothetical scenario"),
    ("scheduled maintenance on payment", "maintenance window"),
    ("payment is under maintenance", "maintenance window"),
])
def test_parser_temporal_uncertainty_and_retraction_rejections(parser, text, expected_phrase):
    r = parser.parse(text)
    assert r.status == AMBIGUOUS
    assert r.incident is None
    assert expected_phrase in r.message

