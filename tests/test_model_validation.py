"""GATE 1 - the system model validates; invalid configurations fail early (master spec §102)."""
import copy
import json

import pytest
from conftest import action, build, hard, raw_model, rule

from ripplecut.errors import ErrorCode, ModelValidationError, RippleCutError
from ripplecut.model.loader import DEFAULT_MODEL_PATH, parse_model
from ripplecut.model.schema import RuleType, Strength
from ripplecut.pipeline import load_scenarios
from ripplecut.solvers.builtin import build_default_registry
from ripplecut.solvers.policy import load_policy, parse_policy


def test_canonical_model_validates(bundle):
    s = bundle.system
    assert len(s.services) == 11
    assert s.c_min == 7
    assert s.critical == {"frontend", "checkoutservice", "paymentservice", "cartservice", "productcatalogservice",
                          "currencyservice", "shippingservice"}
    hard_edges = sum(len(r.hard_upstream) for r in s.rules.values())
    soft_edges = sum(len(r.soft_upstream) for r in s.rules.values())
    assert (hard_edges, soft_edges) == (9, 6)          # ~8-10 meaningful (propagating) dependencies
    assert len(bundle.actions) == 8


def test_every_edge_is_labeled_and_backed(bundle):
    for r in bundle.system.rules.values():
        assert r.semantics == "RIPPLECUT-MODELED"
        for inp in r.inputs:
            assert inp.topology == "SOURCE-BACKED"
            assert "src/" in inp.evidence and "L" in inp.evidence    # file + line reference


def test_email_is_soft_for_checkout(bundle):
    rule_ = bundle.system.rules["checkoutservice"]
    assert "emailservice" in rule_.soft_upstream and "emailservice" not in rule_.hard_upstream


def _raw():
    return json.loads(DEFAULT_MODEL_PATH.read_text())


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["dependency_rules"][0]["inputs"].append({"service": "ghost", "strength": "hard"}), "ghost"),
    (lambda d: d["dependency_rules"][0]["inputs"].append(
        {"service": "checkoutservice", "strength": "hard", "topology": "RIPPLECUT-MODELED"}), "self-dependency"),
    (lambda d: d["interventions"][0].update(cost=-1), "nonnegative"),
    (lambda d: d["interventions"][0].update(targets=["ghostservice"]), "ghostservice"),
    (lambda d: d["interventions"][0]["effect"].update(set_up=["cartservice"]), "must equal"),
    (lambda d: d["action_conflicts"].append({"actions": ["payment_fallback", "nope"],
                                             "resolution": "INVALID_COMBINATION"}), "unknown actions"),
    (lambda d: d["criticality"]["critical"].append("ghost"), "unknown services"),
    (lambda d: d["criticality"]["non_critical"].remove("adservice"), "no criticality"),
    (lambda d: d["dependency_rules"][0].update(semantics="SOURCE-BACKED"), "RIPPLECUT-MODELED"),
    (lambda d: d["objective_policy"].update(type="WEIGHTED"), "LEXICOGRAPHIC"),
    (lambda d: d["dependency_rules"].append(copy.deepcopy(d["dependency_rules"][0])), "duplicate rule"),
    (lambda d: d["dependency_rules"][0]["inputs"][0].update(evidence=""), "without evidence"),
    (lambda d: d["action_conflicts"].pop(), "opposite directions"),          # restart vs suppress email
])
def test_invalid_model_is_rejected(mutate, needle):
    d = _raw()
    mutate(d)
    with pytest.raises(ModelValidationError) as exc:
        parse_model(d)
    assert exc.value.code is ErrorCode.MODEL_VALIDATION_ERROR
    assert needle in str(exc.value)


def test_rule_parameters_validated():
    with pytest.raises(ModelValidationError, match="THRESHOLD"):
        build(services=["a", "b", "c"], rules=[rule("c", "THRESHOLD", hard("a", "b"), threshold=3)])
    with pytest.raises(ModelValidationError, match="not hard inputs"):
        build(services=["a", "b", "c"], rules=[rule("c", "GROUP", hard("a"), groups=[{"members": ["b"], "q": 1}])])
    with pytest.raises(ModelValidationError, match="joint_effect"):
        build(services=["a"], actions=[action("x", 1, up=["a"]), action("y", 1, up=["a"])],
              conflicts=[{"actions": ["x", "y"], "resolution": "DEFINED_JOINT_EFFECT"}])


def test_supported_rule_types_parse():
    b = build(services=["a", "b", "c", "d", "e", "f"], rules=[
        rule("c", "OR", hard("a", "b")), rule("d", "THRESHOLD", hard("a", "b", "c"), threshold=2),
        rule("e", "GROUP", hard("a", "b", "c"), groups=[{"members": ["a", "b"], "q": 1}, {"members": ["c"], "q": 1}]),
        rule("f", "AND", hard("a") + [{"service": "b", "strength": "soft"}])])
    assert b.system.rules["e"].rule_type is RuleType.GROUP
    assert b.system.rules["f"].inputs[1].strength is Strength.SOFT


def test_policy_validation():
    reg = build_default_registry()
    load_policy().validate_against(reg)
    bad = {"solvers": {"enabled": ["branch_and_bound", "nope"], "primary": ["branch_and_bound"]}}
    with pytest.raises(RippleCutError, match="unregistered"):
        parse_policy(bad).validate_against(reg)
    with pytest.raises(RippleCutError, match="validation.required"):
        parse_policy({"solvers": {"enabled": ["exhaustive"], "primary": ["exhaustive"],
                                  "validation": {"required": False}}})
    with pytest.raises(RippleCutError, match="not enabled"):
        parse_policy({"solvers": {"enabled": ["exhaustive"], "primary": ["exhaustive"],
                                  "fallback": ["branch_and_bound"]}}).validate_against(reg)
    with pytest.raises(RippleCutError, match="primary"):
        parse_policy({"solvers": {"enabled": ["exhaustive"], "primary": []}})


def test_scenarios_validate_initial_failures(tmp_path, bundle):
    assert len(load_scenarios(None, bundle)) >= 4
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"scenarios": [{"id": "x", "incident": {"failed_services": ["ghostservice"]}}]}))
    with pytest.raises(RippleCutError) as exc:
        load_scenarios(p, bundle)
    assert exc.value.code is ErrorCode.INVALID_SERVICE
    p.write_text(json.dumps({"scenarios": [{"id": "x", "label": "RCAEval incident",
                                            "incident": {"failed_services": ["cartservice"]}}]}))
    with pytest.raises(RippleCutError, match="RCAEval"):
        load_scenarios(p, bundle)


def test_raw_model_helper_roundtrip():
    b = parse_model(raw_model(services=["a", "b"], rules=[rule("b", "AND", hard("a"))], critical=["b"]))
    assert b.system.c_min == 1 and b.system.service_ids == ("a", "b")
