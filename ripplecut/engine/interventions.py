"""The single RippleCut intervention engine (master spec §20-§22, PDF §7).

A candidate plan B is a subset of the finite action universe A. Its joint
transformation T_B : X -> X is defined only through explicit configuration:

1. Legality (checked BEFORE any simulation):
   * every id is a known action and appears once;
   * every precondition holds on the pre-intervention state x(0);
   * no INVALID_COMBINATION pair is contained in B;
   * no action is part of more than one active DEFINED_JOINT_EFFECT pair
     (a higher-order composition would be undefined, so it is rejected).
2. Effect units: each active DEFINED_JOINT_EFFECT pair contributes its joint
   effect; every other action contributes its own effect.
3. Up(B) = union of set_up, Down(B) = union of set_down over the units.
   Up(B) and Down(B) must be disjoint (guaranteed for legal plans by model
   validation; re-checked defensively).
4. T_B(x)_v = 1 if v in Up(B); 0 if v in Down(B); x_v otherwise.

Every illegality reason in (1) is upward closed: if B is illegal, so is every
superset of B. ``LegalityResult.upward_closed`` records this so that exact
solvers may prune safely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from ..errors import ErrorCode, InterventionError
from ..model.schema import (ActionUniverse, ConflictResolution, Effect, Intervention, Plan, State,
                            SystemModel)


@dataclass(frozen=True)
class LegalityResult:
    legal: bool
    code: Optional[str] = None           # ErrorCode value when illegal
    detail: str = ""
    upward_closed: bool = True           # True => every superset is also illegal
    up: FrozenSet[str] = frozenset()
    down: FrozenSet[str] = frozenset()


def precondition_failures(action: Intervention, system: SystemModel, pre_state: State) -> List[str]:
    out = []
    for p in action.preconditions:
        actual = pre_state[system.index(p.service)]
        if actual != p.state:
            want = "UP" if p.state == 1 else "DOWN"
            out.append(f"{action.id} requires {p.service} {want} in the pre-intervention state")
    return out


def check_plan(system: SystemModel, universe: ActionUniverse, plan: Sequence[str],
               pre_state: State) -> LegalityResult:
    ids = list(plan)
    if any(not isinstance(a, str) for a in ids):
        return LegalityResult(False, ErrorCode.INVALID_ACTION.value, "plan contains a non-string action id")
    unknown = sorted({a for a in ids if universe.get(a) is None})
    if unknown:
        return LegalityResult(False, ErrorCode.INVALID_ACTION.value, f"unknown action(s): {unknown}")
    if len(set(ids)) != len(ids):
        return LegalityResult(False, ErrorCode.INVALID_ACTION.value, "plan lists an action more than once")
    chosen = set(ids)
    for aid in sorted(chosen):
        fails = precondition_failures(universe.get(aid), system, pre_state)
        if fails:
            return LegalityResult(False, ErrorCode.PRECONDITION_UNMET.value, "; ".join(fails))

    active = [c for key, c in universe.conflicts.items() if key <= chosen]
    for c in sorted(active, key=lambda c: sorted(c.actions)):
        if c.resolution is ConflictResolution.INVALID_COMBINATION:
            return LegalityResult(False, ErrorCode.ACTION_CONFLICT.value,
                                  f"INVALID_COMBINATION {sorted(c.actions)}: {c.rationale}")
    joint_pairs = [c for c in active if c.resolution is ConflictResolution.DEFINED_JOINT_EFFECT]
    seen: Set[str] = set()
    for c in joint_pairs:
        overlap = seen & c.actions
        if overlap:
            return LegalityResult(False, ErrorCode.ACTION_CONFLICT.value,
                                  f"action(s) {sorted(overlap)} take part in more than one defined joint effect; "
                                  f"the higher-order composition is not defined")
        seen |= c.actions

    units: List[Effect] = [c.joint_effect for c in joint_pairs]          # type: ignore[misc]
    units += [universe.get(a).effect for a in sorted(chosen - seen)]
    up: Set[str] = set()
    down: Set[str] = set()
    for e in units:
        up |= e.set_up
        down |= e.set_down
    if up & down:
        # Unreachable for validated models; reported, never silently resolved.
        return LegalityResult(False, ErrorCode.ACTION_CONFLICT.value,
                              f"effects set {sorted(up & down)} both UP and DOWN without a declared conflict",
                              upward_closed=False)
    return LegalityResult(True, up=frozenset(up), down=frozenset(down))


def apply_plan(system: SystemModel, universe: ActionUniverse, plan: Sequence[str], state: State,
               pre_state: Optional[State] = None) -> State:
    """T_B(state). Raises InterventionError for an illegal plan (never composes silently)."""
    legality = check_plan(system, universe, plan, state if pre_state is None else pre_state)
    if not legality.legal:
        raise InterventionError(legality.detail, code=ErrorCode(legality.code))
    return transform(system, state, legality.up, legality.down)


def transform(system: SystemModel, state: State, up: Iterable[str], down: Iterable[str]) -> State:
    x = list(state)
    for sid in up:
        x[system.index(sid)] = 1
    for sid in down:
        x[system.index(sid)] = 0
    return tuple(x)


def optimistic_state(system: SystemModel, universe: ActionUniverse, pre_state: State,
                     included: Sequence[str], candidates: Sequence[str]) -> State:
    """An upper bound on T_D(x(0)) for EVERY legal plan D with included <= D <= included + candidates.

    Every service that some unit of such a D could set UP is set to 1; no DOWN
    effect is applied. Candidates whose preconditions fail on x(0) can never be
    part of a legal D and are ignored. Because Phi is monotone, the cascade of
    this state bounds C(D) from above and R(D) from below (see docs/MATHEMATICS.md).
    """
    pool = set(included) | {a for a in candidates
                            if not precondition_failures(universe.get(a), system, pre_state)}
    up: Set[str] = set()
    for aid in pool:
        up |= universe.get(aid).effect.set_up
    for key, c in universe.conflicts.items():
        if c.resolution is ConflictResolution.DEFINED_JOINT_EFFECT and key <= pool and c.joint_effect:
            up |= c.joint_effect.set_up
    x = list(pre_state)
    for sid in up:
        x[system.index(sid)] = 1
    return tuple(x)


def statically_applicable(system: SystemModel, universe: ActionUniverse, pre_state: State) -> Tuple[str, ...]:
    """Actions whose preconditions hold on x(0) (the others can never appear in a legal plan)."""
    return tuple(a.id for a in universe.interventions if not precondition_failures(a, system, pre_state))
