"""Immutable internal schema for RippleCut (master spec §55).

All objects are frozen so that one ``ContainmentProblem`` can be handed to any
number of solvers (possibly on worker threads) without any of them being able
to mutate the shared model.

Formal conventions (FINAL AUDITED PDF §4-§8):

* A Boolean system state is ``x in {0,1}^|V|``; it is stored as a tuple of ints
  aligned with ``SystemModel.service_ids`` (1 = UP, 0 = not UP).
* ``DEGRADED`` exists only in the descriptive / state-estimation layer and is
  mapped to 0 in the formal model because ``x_v = 1 iff v is UP``.
* Costs are exact rationals (``fractions.Fraction``) so that cost ties used by
  the lexicographic objective are never broken by floating-point rounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

State = Tuple[int, ...]
Plan = Tuple[str, ...]          # canonical: sorted, unique action ids

UP = 1
DOWN = 0

SOURCE_BACKED = "SOURCE-BACKED"
RIPPLECUT_MODELED = "RIPPLECUT-MODELED"


def frozen_mapping(data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(data or {}))


class RuleType(str, Enum):
    AND = "AND"
    OR = "OR"
    THRESHOLD = "THRESHOLD"
    GROUP = "GROUP"          # conjunction of threshold groups


class Strength(str, Enum):
    HARD = "hard"            # participates in the Boolean rule R_v
    SOFT = "soft"            # topology only; failure never propagates via this edge


class ConflictResolution(str, Enum):
    INVALID_COMBINATION = "INVALID_COMBINATION"
    DEFINED_JOINT_EFFECT = "DEFINED_JOINT_EFFECT"


class ServiceRole(str, Enum):
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class Service:
    id: str
    name: str
    role: ServiceRole
    description: str
    aliases: Tuple[str, ...]
    source: str


@dataclass(frozen=True)
class DependencyInput:
    service: str
    strength: Strength
    topology: str            # SOURCE-BACKED or RIPPLECUT-MODELED
    evidence: str


@dataclass(frozen=True)
class DependencyGroup:
    members: Tuple[str, ...]
    q: int


@dataclass(frozen=True)
class DependencyRule:
    downstream: str
    rule_type: RuleType
    inputs: Tuple[DependencyInput, ...]
    threshold: Optional[int] = None                 # THRESHOLD only
    groups: Tuple[DependencyGroup, ...] = ()        # GROUP only
    semantics: str = RIPPLECUT_MODELED
    source: str = ""
    notes: str = ""

    @property
    def hard_upstream(self) -> Tuple[str, ...]:
        return tuple(i.service for i in self.inputs if i.strength is Strength.HARD)

    @property
    def soft_upstream(self) -> Tuple[str, ...]:
        return tuple(i.service for i in self.inputs if i.strength is Strength.SOFT)


@dataclass(frozen=True)
class Precondition:
    service: str
    state: int               # UP (1) or DOWN (0), evaluated on the pre-intervention state x(0)


@dataclass(frozen=True)
class Effect:
    set_up: FrozenSet[str] = frozenset()
    set_down: FrozenSet[str] = frozenset()

    @property
    def services(self) -> FrozenSet[str]:
        return self.set_up | self.set_down


@dataclass(frozen=True)
class Intervention:
    id: str
    name: str
    description: str
    cost: Fraction
    targets: Tuple[str, ...]
    preconditions: Tuple[Precondition, ...]
    effect: Effect
    conflicts: Tuple[str, ...]          # derived from ActionUniverse.conflicts (single source of truth)
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)


@dataclass(frozen=True)
class ActionConflict:
    actions: FrozenSet[str]             # exactly two action ids
    resolution: ConflictResolution
    joint_effect: Optional[Effect]
    rationale: str


@dataclass(frozen=True)
class ActionUniverse:
    """The finite action universe A (PDF §7)."""

    interventions: Tuple[Intervention, ...]                 # sorted by id
    conflicts: Mapping[FrozenSet[str], ActionConflict]

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(a.id for a in self.interventions)

    def get(self, action_id: str) -> Optional[Intervention]:
        return self._index.get(action_id)

    @property
    def _index(self) -> Mapping[str, Intervention]:
        idx = self.__dict__.get("_idx_cache")
        if idx is None:
            idx = MappingProxyType({a.id: a for a in self.interventions})
            object.__setattr__(self, "_idx_cache", idx)
        return idx

    def __len__(self) -> int:
        return len(self.interventions)


@dataclass(frozen=True)
class SystemModel:
    model_id: str
    name: str
    services: Tuple[Service, ...]
    rules: Mapping[str, DependencyRule]          # downstream -> rule; services without a rule are roots (R_v = 1)
    critical: FrozenSet[str]
    non_critical: FrozenSet[str]
    infrastructure: FrozenSet[str]
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    @property
    def service_ids(self) -> Tuple[str, ...]:
        return tuple(s.id for s in self.services)

    def index(self, service_id: str) -> int:
        idx = self.__dict__.get("_pos_cache")
        if idx is None:
            idx = {sid: i for i, sid in enumerate(self.service_ids)}
            object.__setattr__(self, "_pos_cache", idx)
        return idx[service_id]

    def has_service(self, service_id: str) -> bool:
        return service_id in self.service_ids

    @property
    def c_min(self) -> int:
        """C_min = sum_v c(v): number of designated critical services."""
        return len(self.critical)

    def all_up(self) -> State:
        return tuple(UP for _ in self.services)

    def state_with_failures(self, failed: Iterable[str]) -> State:
        failed_set = set(failed)
        return tuple(DOWN if sid in failed_set else UP for sid in self.service_ids)

    def state_from_mapping(self, mapping: Mapping[str, int]) -> State:
        return tuple(int(mapping[sid]) for sid in self.service_ids)

    def state_to_mapping(self, state: State) -> Dict[str, int]:
        return {sid: int(state[i]) for i, sid in enumerate(self.service_ids)}

    def down_services(self, state: State) -> Tuple[str, ...]:
        return tuple(sid for i, sid in enumerate(self.service_ids) if state[i] == DOWN)


@dataclass(frozen=True)
class ObjectivePolicy:
    """Frozen lexicographic policy (PDF §8.3). Not configurable in the MVP."""

    kind: str = "LEXICOGRAPHIC"
    order: Tuple[str, ...] = ("critical_feasibility", "total_cost", "intervention_count", "residual_failures")
    final_tie_breaker: str = "sorted_action_ids"


@dataclass(frozen=True)
class ResourceLimits:
    timeout_seconds: float = 10.0
    max_evaluations: int = 200_000
    grace_seconds: float = 0.5


@dataclass(frozen=True)
class ContainmentProblem:
    """Everything a solver needs (master spec §26). Solvers receive nothing else."""

    problem_id: str
    system: SystemModel
    actions: ActionUniverse
    initial_state: State
    objective: ObjectivePolicy
    resource_limits: ResourceLimits
    require_all_critical: bool = True
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    @property
    def m(self) -> int:
        return len(self.actions)


def canonical_plan(action_ids: Iterable[str]) -> Plan:
    """Sorted tuple of ids. Duplicates are preserved so that legality checks can reject them."""
    return tuple(sorted(action_ids))


def cost_to_json(value: Fraction) -> Any:
    value = Fraction(value)
    return int(value) if value.denominator == 1 else float(value)
