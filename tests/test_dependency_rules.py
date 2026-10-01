"""Dependency rules R_v (PDF §6.1): AND, OR, THRESHOLD, GROUP, soft edges; all monotone."""
import itertools

from conftest import build, hard, rule, soft

from ripplecut.engine.dependency import describe_rule, rule_values


def _values(bundle_, **state):
    x = bundle_.system.state_from_mapping({s: state.get(s, 1) for s in bundle_.system.service_ids})
    return rule_values(bundle_.system, x)


def test_and_rule():
    b = build(services=["a", "b", "c"], rules=[rule("c", "AND", hard("a", "b"))])
    assert _values(b)["c"] == 1
    assert _values(b, a=0)["c"] == 0
    assert _values(b, b=0)["c"] == 0


def test_or_rule():
    b = build(services=["a", "b", "c"], rules=[rule("c", "OR", hard("a", "b"))])
    assert _values(b, a=0)["c"] == 1
    assert _values(b, a=0, b=0)["c"] == 0


def test_threshold_rule_2_of_3():
    b = build(services=["a", "b", "c", "d"], rules=[rule("d", "THRESHOLD", hard("a", "b", "c"), threshold=2)])
    assert _values(b, a=0)["d"] == 1
    assert _values(b, a=0, b=0)["d"] == 0


def test_group_rule():
    b = build(services=["db1", "db2", "cache", "api"], rules=[rule("api", "GROUP", hard("db1", "db2", "cache"),
              groups=[{"members": ["db1", "db2"], "q": 1}, {"members": ["cache"], "q": 1}])])
    assert _values(b, db1=0)["api"] == 1
    assert _values(b, db1=0, db2=0)["api"] == 0
    assert _values(b, cache=0)["api"] == 0


def test_soft_dependency_never_propagates():
    b = build(services=["a", "b", "c"], rules=[rule("c", "AND", hard("a") + soft("b"))])
    assert _values(b, b=0)["c"] == 1
    assert _values(b, a=0)["c"] == 0
    assert "soft, non-propagating: b" in describe_rule(b.system.rules["c"])


def test_service_without_rule_is_root():
    b = build(services=["a", "b"], rules=[rule("b", "AND", soft("a"))])
    assert _values(b, a=0) == {"a": 1, "b": 1}          # R_v for roots and soft-only rules is 1


def test_every_rule_type_is_monotone():
    b = build(services=["a", "b", "c", "o", "t", "g", "n"], rules=[
        rule("o", "OR", hard("a", "b")), rule("t", "THRESHOLD", hard("a", "b", "c"), threshold=2),
        rule("g", "GROUP", hard("a", "b", "c"), groups=[{"members": ["a", "b"], "q": 1}, {"members": ["c"], "q": 1}]),
        rule("n", "AND", hard("a", "b", "c"))])
    ids = b.system.service_ids
    states = list(itertools.product((0, 1), repeat=len(ids)))
    for x in states:
        rx = rule_values(b.system, x)
        for i in range(len(ids)):
            if x[i] == 0:
                y = x[:i] + (1,) + x[i + 1:]
                ry = rule_values(b.system, y)
                assert all(rx[s] <= ry[s] for s in ids)
