"""Structured, explicit error types for RippleCut (master spec §101).

Every error carries a machine-readable ``ErrorCode`` so that failures are never
swallowed silently and can be surfaced in logs, the CLI and the UI.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    MODEL_VALIDATION_ERROR = "MODEL_VALIDATION_ERROR"
    INVALID_SERVICE = "INVALID_SERVICE"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    INVALID_ACTION = "INVALID_ACTION"
    ACTION_CONFLICT = "ACTION_CONFLICT"
    PRECONDITION_UNMET = "PRECONDITION_UNMET"
    SIMULATION_ERROR = "SIMULATION_ERROR"
    SOLVER_ERROR = "SOLVER_ERROR"
    SOLVER_TIMEOUT = "SOLVER_TIMEOUT"
    SOLVER_RESOURCE_LIMIT = "SOLVER_RESOURCE_LIMIT"
    SOLVER_INCOMPATIBLE = "SOLVER_INCOMPATIBLE"
    SOLVER_INVALID_RESULT = "SOLVER_INVALID_RESULT"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    NO_VALID_SOLVER_RESULT = "NO_VALID_SOLVER_RESULT"
    LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
    INCIDENT_AMBIGUOUS = "INCIDENT_AMBIGUOUS"
    DATA_ADAPTER_ERROR = "DATA_ADAPTER_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"


class RippleCutError(Exception):
    """Base class. ``code`` is always set; ``details`` is JSON-serialisable."""

    default_code = ErrorCode.CONFIG_ERROR

    def __init__(self, message: str, code: Optional[ErrorCode] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code.value}] {self.message}"


class ModelValidationError(RippleCutError):
    default_code = ErrorCode.MODEL_VALIDATION_ERROR


class SimulationError(RippleCutError):
    default_code = ErrorCode.SIMULATION_ERROR


class InterventionError(RippleCutError):
    default_code = ErrorCode.INVALID_ACTION


class IncidentError(RippleCutError):
    default_code = ErrorCode.INVALID_SERVICE


class DataAdapterError(RippleCutError):
    default_code = ErrorCode.DATA_ADAPTER_ERROR


class SolverInterrupt(RippleCutError):
    """Raised *inside* a solver by its context when a limit is hit.

    Solvers may catch it to return an incumbent; if they do not, the
    execution guard converts it into a TIMEOUT / RESOURCE_LIMIT result.
    """


class SolverTimeout(SolverInterrupt):
    default_code = ErrorCode.SOLVER_TIMEOUT


class ResourceLimitExceeded(SolverInterrupt):
    default_code = ErrorCode.SOLVER_RESOURCE_LIMIT
