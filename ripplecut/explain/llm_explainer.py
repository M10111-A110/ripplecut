"""Optional LLM phrasing of deterministic facts, with a post-generation guard (master spec §71).

The LLM receives ONLY the structured facts produced by the deterministic
explainer and may rephrase them. Its text is accepted only if the guard finds
no issue:

  * every service-like token names a modeled service;
  * every action-like token names a configured action that appears in the facts;
  * every "cost <number>" matches a cost present in the facts;
  * no claim of automatic execution / remediation / production readiness.

On any issue (or any LLM error) the deterministic text is used and the issues
are reported. The guard is conservative by design: false rejections only cost
fluency, never correctness.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional

from ..model.schema import ActionUniverse, SystemModel

_SERVICE_TOKEN = re.compile(r"\b[a-z][a-z0-9-]*service\b")
_ACTION_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_COST = re.compile(r"\bcosts?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
FORBIDDEN = ("automatically fixed", "automatically fixes", "has been applied", "was executed", "has been executed",
             "production-ready", "production ready", "guaranteed in production")


def check_explanation(text: str, facts: Mapping[str, Any], system: SystemModel, actions: ActionUniverse) -> List[str]:
    issues: List[str] = []
    low = text.lower()
    fact_blob = json.dumps(facts, default=str)
    for tok in sorted(set(_SERVICE_TOKEN.findall(low))):
        if not system.has_service(tok):
            issues.append(f"unknown service '{tok}'")
    for tok in sorted(set(_ACTION_TOKEN.findall(low))):
        if actions.get(tok) is None:
            issues.append(f"unknown action '{tok}'")
        elif tok not in fact_blob:
            issues.append(f"action '{tok}' is not part of the provided facts")
    fact_costs = {str(v) for v in re.findall(r'"(?:cost|K)": (\d+(?:\.\d+)?)', fact_blob)}
    for c in _COST.findall(text):
        norm = str(int(float(c))) if float(c).is_integer() else c
        if norm not in fact_costs:
            issues.append(f"cost {c} does not appear in the facts")
    for phrase in FORBIDDEN:
        if phrase in low:
            issues.append(f"forbidden claim '{phrase}'")
    return issues


def llm_explanation(client: Any, facts: Mapping[str, Any], system: SystemModel,
                    actions: ActionUniverse) -> Dict[str, Any]:
    prompt_facts = {k: facts[k] for k in ("initial_failures", "cascade", "uncontrolled_critical_down", "status",
                                          "selected_plan", "objective", "runner_ups", "optimality", "verification",
                                          "infeasibility", "approval") if k in facts}
    system_prompt = ("Rephrase the given JSON facts about a modeled incident as a short explanation for an on-call "
                     "engineer. Use only the facts. Do not add services, actions, costs, dependencies or outcomes. "
                     "Do not claim anything was executed; a human must approve.")
    try:
        text = client.complete(system_prompt, json.dumps(prompt_facts, default=str))
    except Exception as exc:  # noqa: BLE001 - LLM is optional; fall back and report
        return {"accepted": False, "text": None, "issues": [f"LLM unavailable: {type(exc).__name__}"]}
    issues = check_explanation(text, prompt_facts, system, actions)
    return {"accepted": not issues, "text": text if not issues else None, "rejected_text": text if issues else None,
            "issues": issues}
