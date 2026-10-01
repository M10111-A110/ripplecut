"""GATE 4 - objective evaluator on hand-calculated examples (PDF §8; master spec §115)."""
from fractions import Fraction

from conftest import action, build, make_problem

from ripplecut.engine.objective import evaluate_plan, objective_key


def lexi_bundle():
    """Master spec §115. C1 critical; N1, N2 non-critical; all three initially DOWN.

    a0 cost 0: N2 UP            a1 cost 1: N1 UP (does not help C1)
    a2 cost 2: C1 UP            a3 cost 2: C1 and N2 UP
    """
    return build(services=["C1", "N1", "N2"], critical=["C1"], actions=[
        action("a0", 0, up=["N2"]), action("a1", 1, up=["N1"]), action("a2", 2, up=["C1"]),
        action("a3", 2, up=["C1", "N2"])])


def test_hand_calculated_values():
    b = lexi_bundle()
    p = make_problem(b, ["C1", "N1", "N2"])
    want = {(): (False, 0, 0, 3), ("a1",): (False, 1, 1, 2), ("a2",): (True, 2, 1, 2), ("a3",): (True, 2, 1, 1),
            ("a0", "a2"): (True, 2, 2, 1), ("a1", "a3"): (True, 3, 2, 0)}
    for plan, (feasible, k, n, r) in want.items():
        o = evaluate_plan(p, plan).objective
        assert (o.feasible, o.cost, o.count, o.residual) == (feasible, k, n, r), plan


def test_exact_lexicographic_order():
    b = lexi_bundle()
    p = make_problem(b, ["C1", "N1", "N2"])
    ev = {plan: evaluate_plan(p, plan) for plan in [("a1",), ("a2",), ("a3",), ("a0", "a2"), ("a1", "a3")]}
    assert not ev[("a1",)].feasible                        # cheaper (K=1) but fails critical preservation
    assert ev[("a3",)].key < ev[("a0", "a2")].key            # same K=2: fewer actions wins
    assert ev[("a3",)].key < ev[("a2",)].key                 # same K, N: fewer residual failures wins
    assert ev[("a3",)].key < ev[("a1", "a3")].key            # R=0 does not beat a lower cost
    feasible = sorted((e for e in ev.values() if e.feasible), key=lambda e: e.key)
    assert feasible[0].plan == ("a3",)


def test_no_weighted_score_trap():
    """A weighted score such as 10*R + K would pick {a1, a3} (score 3) over {a3} (score 12). RippleCut must not."""
    b = lexi_bundle()
    p = make_problem(b, ["C1", "N1", "N2"])
    best, trap = evaluate_plan(p, ["a3"]), evaluate_plan(p, ["a1", "a3"])
    assert 10 * trap.objective.residual + trap.objective.cost < 10 * best.objective.residual + best.objective.cost
    assert best.key < trap.key


def test_final_tie_break_is_sorted_ids():
    b = build(services=["C"], critical=["C"], actions=[action("zeta", 1, up=["C"]), action("alpha", 1, up=["C"])])
    p = make_problem(b, ["C"])
    assert evaluate_plan(p, ["alpha"]).key < evaluate_plan(p, ["zeta"]).key


def test_costs_are_exact_rationals():
    b = build(services=["C"], critical=["C"], actions=[action("x", 0.1, up=["C"]), action("y", 0.2, up=["C"]),
                                                          action("z", 0.3, up=["C"])])
    p = make_problem(b, ["C"])
    xy = evaluate_plan(p, ["x", "y"]).objective.cost
    assert xy == Fraction(3, 10) == evaluate_plan(p, ["z"]).objective.cost   # 0.1 + 0.2 == 0.3 exactly


def test_residual_counts_all_services_and_infeasibility(bundle):
    p = make_problem(bundle, ["paymentservice", "emailservice"])
    o = evaluate_plan(p, ["payment_fallback"]).objective
    assert o.feasible and o.residual == 1 and o.residual_services == ("emailservice",)
    empty = evaluate_plan(p, []).objective
    assert not empty.feasible and set(empty.critical_down) == {"checkoutservice", "paymentservice"}
    assert empty.critical_up == 5 and empty.critical_required == 7


def test_objective_key_shape(bundle):
    p = make_problem(bundle, ["paymentservice"])
    ev = evaluate_plan(p, ["payment_fallback"])
    assert objective_key(ev.objective, ev.plan) == (Fraction(3), 1, 0, ("payment_fallback",))
