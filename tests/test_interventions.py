"""GATE 3 - intervention engine: legality before simulation, explicit composition, T_B (PDF §7)."""
import itertools
import random

import pytest
from conftest import action, build, hard, make_problem, rule

from ripplecut.engine.interventions import apply_plan, check_plan, optimistic_state, transform
from ripplecut.engine.objective import evaluate_plan
from ripplecut.errors import ErrorCode, InterventionError
from ripplecut.generators import random_problem


def _x0(bundle, *failed):
    return bundle.system.state_with_failures(failed)


def test_single_action(bundle):
    x0 = _x0(bundle, "paymentservice")
    x = apply_plan(bundle.system, bundle.actions, ["payment_fallback"], x0)
    assert bundle.system.state_to_mapping(x)["paymentservice"] == 1


def test_multiple_actions(bundle):
    x0 = _x0(bundle, "paymentservice", "shippingservice")
    x = apply_plan(bundle.system, bundle.actions, ["payment_fallback", "shipping_fallback"], x0)
    assert bundle.system.down_services(x) == ()


def test_invalid_combination_rejected_before_simulation(bundle):
    p = make_problem(bundle, ["paymentservice"])
    ev = evaluate_plan(p, ["checkout_backend_standby", "payment_fallback"])
    assert not ev.legal and ev.legality.code == ErrorCode.ACTION_CONFLICT.value
    assert ev.cascade is None and ev.objective is None          # no simulation happened
    assert ev.legality.upward_closed


def test_unknown_and_duplicate_actions(bundle):
    x0 = _x0(bundle, "paymentservice")
    r = check_plan(bundle.system, bundle.actions, ["reboot_everything"], x0)
    assert not r.legal and r.code == ErrorCode.INVALID_ACTION.value
    r = check_plan(bundle.system, bundle.actions, ["payment_fallback", "payment_fallback"], x0)
    assert not r.legal and r.code == ErrorCode.INVALID_ACTION.value
    with pytest.raises(InterventionError):
        apply_plan(bundle.system, bundle.actions, ["reboot_everything"], x0)


def test_precondition_evaluated_on_pre_intervention_state(bundle):
    r = check_plan(bundle.system, bundle.actions, ["payment_fallback"], bundle.system.all_up())
    assert not r.legal and r.code == ErrorCode.PRECONDITION_UNMET.value
    r = check_plan(bundle.system, bundle.actions, ["restart_checkoutservice"], _x0(bundle, "paymentservice"))
    assert not r.legal     # checkout is UP in x(0): it only fails later in the cascade


def test_defined_joint_effect():
    b = build(
        services=["a", "b"],
        actions=[
            action("restart_a", 1, up=["a"], pre=[("a", "DOWN")]),
            action("suppress_a", 0, down=["a"]),
        ],
        conflicts=[{
            "actions": ["restart_a", "suppress_a"],
            "resolution": "DEFINED_JOINT_EFFECT",
            "joint_effect": {"set_up": [], "set_down": ["a"]},
            "rationale": "explicit policy: suppression wins",
        }],
    )
    x0 = b.system.state_with_failures(["a"])
    r = check_plan(b.system, b.actions, ["restart_a", "suppress_a"], x0)
    assert r.legal and r.up == frozenset() and r.down == {"a"}   # explicit policy: suppression wins
    x = apply_plan(b.system, b.actions, ["restart_a", "suppress_a"], x0)
    assert b.system.state_to_mapping(x)["a"] == 0


def test_action_in_two_joint_pairs_is_illegal():
    b = build(services=["a", "b"], actions=[action("x", 1, up=["a"]), action("y", 1, down=["a"]),
                                           action("z", 1, down=["a"])],
              conflicts=[{"actions": ["x", "y"], "resolution": "DEFINED_JOINT_EFFECT", "joint_effect": {"set_down": ["a"]}},
                         {"actions": ["x", "z"], "resolution": "DEFINED_JOINT_EFFECT", "joint_effect": {"set_down": ["a"]}}])
    r = check_plan(b.system, b.actions, ["x", "y", "z"], b.system.all_up())
    assert not r.legal and "more than one defined joint effect" in r.detail


def test_deterministic_effects(bundle):
    x0 = _x0(bundle, "paymentservice", "shippingservice")
    a = apply_plan(bundle.system, bundle.actions, ["checkout_backend_standby"], x0)
    assert a == apply_plan(bundle.system, bundle.actions, ["checkout_backend_standby"], x0)
    assert transform(bundle.system, x0, {"paymentservice"}, set()) != x0


def test_tiny_abc_intervention_restores_everything():
    """Master spec §114, second half: an intervention restoring A gives A=B=C=UP."""
    b = build(services=["A", "B", "C"], rules=[rule("C", "AND", hard("A", "B"))], critical=["C"],
              actions=[action("restore_A", 1, up=["A"], pre=[("A", "DOWN")])])
    p = make_problem(b, state={"A": 0, "B": 1, "C": 1})
    ev = evaluate_plan(p, ["restore_A"])
    assert b.system.state_to_mapping(ev.cascade.final_state) == {"A": 1, "B": 1, "C": 1}
    assert ev.feasible and ev.objective.residual == 0


@pytest.mark.parametrize("seed", range(40))
def test_illegality_is_upward_closed_random(seed):
    """Every illegality reason is upward closed (B&B's S1 prune relies on it)."""
    problem, b = random_problem(seed, n_services=6, m_actions=6)
    ids = sorted(b.actions.ids)
    for r in range(len(ids) + 1):
        for combo in itertools.combinations(ids, r):
            res = check_plan(b.system, b.actions, combo, problem.initial_state)
            if not res.legal:
                assert res.upward_closed
                for extra in set(ids) - set(combo):
                    assert not check_plan(b.system, b.actions, combo + (extra,), problem.initial_state).legal


@pytest.mark.parametrize("seed", range(40))
def test_optimistic_state_dominates_every_legal_completion(seed):
    """T_D(x0) <= x_opt for every legal D with I <= D <= I + candidates (basis of bounds S3/S4)."""
    problem, b = random_problem(seed, n_services=6, m_actions=6)
    rng = random.Random(seed)
    ids = sorted(b.actions.ids)
    included = tuple(rng.sample(ids, rng.randint(0, 2)))
    cands = [a for a in ids if a not in included]
    opt = optimistic_state(b.system, b.actions, problem.initial_state, included, cands)
    for r in range(len(cands) + 1):
        for extra in itertools.combinations(cands, r):
            res = check_plan(b.system, b.actions, included + extra, problem.initial_state)
            if res.legal:
                td = transform(b.system, problem.initial_state, res.up, res.down)
                assert all(a <= o for a, o in zip(td, opt))
