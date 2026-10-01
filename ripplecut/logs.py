"""Structured JSON logging (master spec §66).

``log_event`` emits one JSON object per event on the ``ripplecut`` logger.
Nothing secret is ever logged: API keys are read from the environment by the
optional LLM client and never passed to this module.

``EventRecorder`` is a logging handler that keeps the structured events of one
run in memory so they can be attached to the run report and shown in the UI.
"""
from __future__ import annotations

import json
import logging
import threading
from fractions import Fraction
from typing import Any, Dict, List

LOGGER_NAME = "ripplecut"


def _default(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return int(obj) if obj.denominator == 1 else float(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=_default, sort_keys=True),
                extra={"ripplecut_event": json.loads(json.dumps(payload, default=_default))})


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False          # RippleCut handlers decide what is printed
    return logger


class EventRecorder(logging.Handler):
    """Collects the structured events emitted through ``log_event`` by ONE thread.

    Filtering on the creating thread keeps concurrent runs (e.g. two UI
    requests) from mixing their event streams.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: List[Dict[str, Any]] = []
        self._thread = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        ev = getattr(record, "ripplecut_event", None)
        if ev is not None and record.thread == self._thread:
            self.events.append(ev)

    def __enter__(self) -> "EventRecorder":
        logger = get_logger()
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        get_logger().removeHandler(self)


def configure_cli_logging(verbose: bool) -> None:
    """JSON lines to stderr when verbose; otherwise only warnings."""
    logger = get_logger()
    if not any(getattr(h, "_ripplecut_cli", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._ripplecut_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    for h in logger.handlers:
        if getattr(h, "_ripplecut_cli", False):
            h.setLevel(logging.INFO if verbose else logging.WARNING)
