"""GATE 2 - cascade simulator: Phi(x)_v = x_v AND R_v(x), synchronous, to a fixed point (PDF §5)."""
import itertools
import random

import pytest
from conftest import build, hard, rule, soft

from ripplecut.engine.simulator import FIXED_POINT, ROUND_LIMIT, cascade_fixed_point, phi, simulate
from ripplecut.errors import SimulationError
from ripplecut.generators import random_problem


def test_critical_correctness_tiny_abc():
    """Master spec §114: C = A AND B; A DOWN -> after cascade C DOWN."""
    b = build(services=["A", "B", "C"], rules=[rule("C", "AND", hard("A", "B"))], critical=["C"])
    x0 = b.system.state_from_mapping({"A": 0, "B": 1, "C": 1})
    res = simulate(b.system, x0)
    assert b.system.state_to_mapping(res.final_state) == {"A": 0, "B": 1, "C": 0}
    assert res.fixed_point and res.termination_reason == FIXED_POINT
    assert res.history[1].newly_failed == ("C",)


def test_one_round_and_multi_round_chain():
    b = build(services=["a", "b", "c", "d"], rules=[rule("b", "AND", hard("a")), rule("c", "AND", hard("b")),
                                                     rule("d", "AND", hard("c"))])
    res = simulate(b.system, b.system.state_with_failures(["a"]))
    assert [r.newly_failed for r in res.history[1:]] == [("b",), ("c",), ("d",)]
    assert res.state_changing_rounds == 3 and res.phi_evaluations == 4
    one = simulate(b.system, b.system.state_with_failures(["c"]))
    assert one.state_changing_rounds == 1 and one.history[1].newly_failed == ("d",)


def test_already_stable_and_no_failures(bundle):
    x = bundle.system.all_up()
    res = simulate(bundle.system, x)
    assert res.final_state == x and res.state_changing_rounds == 0 and res.phi_evaluations == 1
    email = bundle.system.state_with_failures(["emailservice"])
    assert simulate(bundle.system, email).final_state == email      # soft edge: already a fixed point


def test_multiple_initial_failures(bundle):
    res = simulate(bundle.system, bundle.system.state_with_failures(["paymentservice", "shippingservice"]))
    assert bundle.system.down_services(res.final_state) == ("checkoutservice", "paymentservice", "shippingservice")


def test_canonical_catalog_fan_out(bundle):
    res = simulate(bundle.system, bundle.system.state_with_failures(["productcatalogservice"]))
    assert set(res.history[1].newly_failed) == {"checkoutservice", "frontend", "recommendationservice"}


def test_no_spontaneous_recovery():
    """A DOWN service with every input UP stays DOWN during the uncontrolled cascade."""
    b = build(services=["a", "b"], rules=[rule("b", "AND", hard("a"))])
    x0 = b.system.state_from_mapping({"a": 1, "b": 0})
    assert simulate(b.system, x0).final_state == x0


def test_round_limit_protection():
    b = build(services=["a", "b", "c"], rules=[rule("b", "AND", hard("a")), rule("c", "AND", hard("b"))])
    res = simulate(b.system, b.system.state_with_failures(["a"]), max_rounds=1)
    assert not res.fixed_point and res.termination_reason == ROUND_LIMIT


def test_cascade_fixed_point_raises_instead_of_looping(monkeypatch):
    b = build(services=["a", "b", "c"], rules=[rule("b", "AND", hard("a")), rule("c", "AND", hard("b"))])
    import ripplecut.engine.simulator as sim
    orig = sim.simulate
    monkeypatch.setattr(sim, "simulate", lambda s, x, max_rounds=None: orig(s, x, max_rounds=1))
    with pytest.raises(SimulationError):
        sim.cascade_fixed_point(b.system, b.system.state_with_failures(["a"]))


def test_dependency_cycle_terminates():
    b = build(services=["a", "b", "c"], rules=[rule("a", "AND", hard("b")), rule("b", "AND", hard("a")),
                                                rule("c", "OR", hard("a", "b"))])
    res = simulate(b.system, b.system.state_with_failures(["a"]))
    assert res.fixed_point and b.system.down_services(res.final_state) == ("a", "b", "c")
    assert simulate(b.system, b.system.all_up()).final_state == b.system.all_up()


def test_empty_dependencies_and_isolated_service():
    b = build(services=["x", "y", "z"], rules=[rule("y", "AND", soft("x"))])
    res = simulate(b.system, b.system.state_with_failures(["x"]))
    assert b.system.down_services(res.final_state) == ("x",)


def test_invalid_state_rejected(bundle):
    with pytest.raises(SimulationError):
        simulate(bundle.system, (1, 0))
    with pytest.raises(SimulationError):
        simulate(bundle.system, tuple([2] * 11))


@pytest.mark.parametrize("seed", range(60))
def test_termination_descent_and_monotonicity_random(seed):
    """Property tests (master spec §63): terminate within |V|+1, Phi(x) <= x, x <= y => Phi(x) <= Phi(y)."""
    problem, b = random_problem(seed, n_services=7, m_actions=3)
    s, n = b.system, len(b.system.services)
    rng = random.Random(seed)
    for _ in range(20):
        x = tuple(rng.randint(0, 1) for _ in range(n))
        res = simulate(s, x)
        assert res.fixed_point and res.phi_evaluations <= n + 1 and res.state_changing_rounds <= n
        px = phi(s, x)
        assert all(a <= b_ for a, b_ in zip(px, x))
        y = tuple(max(a, rng.randint(0, 1)) for a in x)          # y >= x
        assert all(a <= b_ for a, b_ in zip(phi(s, x), phi(s, y)))
        assert all(a <= b_ for a, b_ in zip(cascade_fixed_point(s, x), cascade_fixed_point(s, y)))


def test_failure_order_does_not_matter(bundle):
    """Master spec §47: the same failure *set* gives the same fixed point, whatever order it is listed in."""
    for perm in itertools.permutations(["paymentservice", "shippingservice", "adservice"]):
        assert simulate(bundle.system, bundle.system.state_with_failures(perm)).final_state == \
            simulate(bundle.system, bundle.system.state_with_failures(sorted(perm))).final_state
