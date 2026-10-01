"""Solver policy (master spec §30, §33): solver selection is configuration, not architecture."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..errors import ErrorCode, RippleCutError
from ..model.schema import ResourceLimits
from .registry import SolverRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "solver_policy.json"


@dataclass(frozen=True)
class SolverPolicy:
    enabled: Tuple[str, ...]
    primary: Tuple[str, ...]
    fallback: Tuple[str, ...]
    limits: ResourceLimits = ResourceLimits()
    accept_uncertified_infeasibility_from_exact_solver: bool = True
    accept_validated_incumbent_if_all_fail: bool = False
    cross_check_solvers: Tuple[str, ...] = ()
    cross_check_max_actions: int = 12
    notes: str = ""

    def execution_order(self) -> Tuple[str, ...]:
        """Primary solvers then fallbacks, de-duplicated, restricted to enabled solvers."""
        out = []
        for name in self.primary + self.fallback:
            if name in self.enabled and name not in out:
                out.append(name)
        return tuple(out)

    def with_timeout(self, seconds: float) -> "SolverPolicy":
        return replace(self, limits=replace(self.limits, timeout_seconds=float(seconds)))

    def validate_against(self, registry: SolverRegistry) -> None:
        """Fail early: every configured solver name must be registered (master spec §102)."""
        missing = sorted({n for n in self.enabled + self.primary + self.fallback + self.cross_check_solvers
                          if n not in registry})
        if missing:
            raise RippleCutError(f"solver policy references unregistered solvers {missing}; "
                                 f"available: {list(registry.list_available())}", ErrorCode.CONFIG_ERROR)
        not_enabled = sorted(set(self.primary + self.fallback) - set(self.enabled))
        if not_enabled:
            raise RippleCutError(f"primary/fallback solvers {not_enabled} are not enabled", ErrorCode.CONFIG_ERROR)
        if not self.execution_order():
            raise RippleCutError("solver policy yields an empty execution order", ErrorCode.CONFIG_ERROR)

    def to_dict(self) -> Mapping[str, Any]:
        return {"enabled": list(self.enabled), "primary": list(self.primary), "fallback": list(self.fallback),
                "execution_order": list(self.execution_order()),
                "limits": {"timeout_seconds": self.limits.timeout_seconds,
                           "max_evaluations": self.limits.max_evaluations,
                           "grace_seconds": self.limits.grace_seconds},
                "accept_uncertified_infeasibility_from_exact_solver":
                    self.accept_uncertified_infeasibility_from_exact_solver,
                "accept_validated_incumbent_if_all_fail": self.accept_validated_incumbent_if_all_fail,
                "cross_check_solvers": list(self.cross_check_solvers),
                "cross_check_max_actions": self.cross_check_max_actions}


def parse_policy(raw: Mapping[str, Any]) -> SolverPolicy:
    s = raw.get("solvers")
    if not isinstance(s, Mapping):
        raise RippleCutError("solver policy must contain a 'solvers' object", ErrorCode.CONFIG_ERROR)

    def names(key: str, required: bool = True) -> Tuple[str, ...]:
        v = s.get(key, [] if not required else None)
        if v is None or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise RippleCutError(f"solvers.{key} must be a list of solver names", ErrorCode.CONFIG_ERROR)
        return tuple(v)

    validation = s.get("validation", {})
    if validation.get("required", True) is not True:
        raise RippleCutError("solvers.validation.required must be true: every final result must be "
                             "independently validated (master spec §56)", ErrorCode.CONFIG_ERROR)
    limits_raw = s.get("limits", {})
    limits = ResourceLimits(timeout_seconds=float(limits_raw.get("timeout_seconds", 10)),
                            max_evaluations=int(limits_raw.get("max_evaluations", 200_000)),
                            grace_seconds=float(limits_raw.get("grace_seconds", 0.5)))
    if limits.timeout_seconds <= 0 or limits.max_evaluations <= 0 or limits.grace_seconds < 0:
        raise RippleCutError("solver limits must be positive", ErrorCode.CONFIG_ERROR)
    verification = s.get("verification", {})
    timeout_policy = s.get("timeout_policy", {})
    policy = SolverPolicy(
        enabled=names("enabled"), primary=names("primary"), fallback=names("fallback", required=False),
        limits=limits,
        accept_uncertified_infeasibility_from_exact_solver=bool(
            validation.get("accept_uncertified_infeasibility_from_exact_solver", True)),
        accept_validated_incumbent_if_all_fail=bool(timeout_policy.get("accept_validated_incumbent_if_all_fail", False)),
        cross_check_solvers=tuple(verification.get("cross_check_solvers", [])),
        cross_check_max_actions=int(verification.get("max_actions", 12)),
        notes=str(s.get("notes", "")),
    )
    if not policy.primary:
        raise RippleCutError("solvers.primary must name at least one solver", ErrorCode.CONFIG_ERROR)
    return policy


def load_policy(path: Optional[Path] = None) -> SolverPolicy:
    p = Path(path) if path else DEFAULT_POLICY_PATH
    try:
        return parse_policy(json.loads(p.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise RippleCutError(f"solver policy not found: {p}", ErrorCode.CONFIG_ERROR) from exc
    except json.JSONDecodeError as exc:
        raise RippleCutError(f"solver policy is not valid JSON: {exc}", ErrorCode.CONFIG_ERROR) from exc
