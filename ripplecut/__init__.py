"""RippleCut — AI-assisted containment planner for modeled cascading failures.

Deterministic core: explicit system model -> cascade simulator (Phi) ->
intervention engine (T_B) -> objective evaluator -> pluggable solvers behind an
execution guard -> independent safety validator -> explanation -> human approval.

The LLM (optional) only parses incident text and phrases explanations; every
LLM output is schema-validated and never reaches the decision core unchecked.
"""
__version__ = "1.0.0"
