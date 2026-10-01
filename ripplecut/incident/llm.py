"""Optional LLM integration (master spec §7, §71, §90, §103, §105).

The LLM is an interface layer only. It may propose structured incident fields
and phrase explanations. It never sees or edits configuration, never proposes
actions, and its output is untrusted input:

    NL text -> LLM -> JSON extraction -> strict schema validation -> StructuredIncident

Any deviation (non-JSON, extra keys such as proposed actions, unknown service
ids, contradictory lists) is rejected with an explicit status. When no API key
is configured or the API call fails, RippleCut falls back to the deterministic
rule-based parser; the core engine never depends on network availability.

The HTTP client uses only the Python standard library. The API key is read
from the ANTHROPIC_API_KEY environment variable and is never logged.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Protocol

from ..errors import ErrorCode, RippleCutError
from ..model.schema import SystemModel
from .parser import AMBIGUOUS, LLM_PARSE_ERROR, OK, UNKNOWN_SERVICE, ParseResult, finalize_candidate

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
LLM_KEYS = frozenset({"failed_services", "degraded_services", "unknown_mentions", "ambiguous", "ambiguity_reason"})


class LLMUnavailable(RippleCutError):
    default_code = ErrorCode.LLM_PARSE_ERROR


class LLMClient(Protocol):
    name: str

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class AnthropicClient:
    """Minimal Messages API client (stdlib only). Construct only when a key is configured."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 20.0) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set; LLM features are disabled (offline mode)")
        self._key = key
        self.model = model or os.environ.get("RIPPLECUT_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({"model": self.model, "max_tokens": 600, "system": system_prompt,
                           "messages": [{"role": "user", "content": user_prompt}]}).encode("utf-8")
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": self._key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"LLM request failed: {type(exc).__name__}") from None
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def optional_client() -> Optional[LLMClient]:
    """An AnthropicClient if a key is configured, else None (offline default)."""
    try:
        return AnthropicClient()
    except LLMUnavailable:
        return None


def extract_json_object(text: str) -> Any:
    """Parse a JSON object from an LLM reply (tolerates ```json fences, nothing else)."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    return json.loads(cleaned)


class LLMIncidentParser:
    name = "llm_parser"

    def __init__(self, system: SystemModel, client: LLMClient) -> None:
        self.system = system
        self.client = client

    def prompts(self, text: str) -> Dict[str, str]:
        services = ", ".join(self.system.service_ids)
        system_prompt = (
            "You convert an incident description into JSON for a deterministic planner. "
            "Reply with ONE JSON object and nothing else, with exactly these keys: "
            '"failed_services" (list of service ids reported DOWN), "degraded_services" (list of ids reported '
            'slow/degraded), "unknown_mentions" (list of components mentioned that are not in the allowed list), '
            '"ambiguous" (true if any service state is unclear), "ambiguity_reason" (string). '
            f"Allowed service ids: {services}. Use only these exact ids. Never add other keys, actions, "
            "dependencies, costs or predictions. If unsure, set ambiguous to true instead of guessing.")
        return {"system": system_prompt, "user": f"Incident: {text}"}

    def parse(self, text: str) -> ParseResult:
        result = ParseResult(OK, self.name, text)
        p = self.prompts(text)
        reply = self.client.complete(p["system"], p["user"])          # LLMUnavailable propagates to caller
        try:
            data = extract_json_object(reply)
        except (json.JSONDecodeError, ValueError):
            result.status, result.message = LLM_PARSE_ERROR, "LLM reply is not a JSON object"
            return result
        if not isinstance(data, dict):
            result.status, result.message = LLM_PARSE_ERROR, "LLM reply is not a JSON object"
            return result
        extra = sorted(set(data) - LLM_KEYS)
        if extra:
            result.status = LLM_PARSE_ERROR
            result.message = f"LLM reply contains unsupported field(s) {extra}; rejected before the engine"
            return result
        for key in ("failed_services", "degraded_services", "unknown_mentions"):
            v = data.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                result.status, result.message = LLM_PARSE_ERROR, f"'{key}' must be a list of strings"
                return result
        if not isinstance(data.get("ambiguous", False), bool):
            result.status, result.message = LLM_PARSE_ERROR, "'ambiguous' must be a boolean"
            return result
        if data.get("unknown_mentions"):
            result.status = UNKNOWN_SERVICE
            result.unknown_terms = sorted(set(data["unknown_mentions"]))
            result.message = f"the incident mentions component(s) not in the model: {result.unknown_terms}"
            return result
        proposed = list(data.get("failed_services", [])) + list(data.get("degraded_services", []))
        bad = sorted({s for s in proposed if not self.system.has_service(s)})
        if bad:
            result.status = UNKNOWN_SERVICE
            result.unknown_terms = bad
            result.message = f"LLM proposed service id(s) that are not in the model: {bad}; rejected"
            return result
        if data.get("ambiguous"):
            result.status = AMBIGUOUS
            result.message = f"LLM reports ambiguity: {data.get('ambiguity_reason', '')}".strip()
            return result
        return finalize_candidate(self.system, result, sorted(set(data.get("failed_services", []))),
                                  sorted(set(data.get("degraded_services", []))), self.name)
