# RippleCut

Containment planner for modeled cascading failures. Team 3 Idiots.

Give RippleCut an incident such as "paymentservice is down". It then:

- simulates how the failure spreads through an explicit model of Google Online Boutique;
- searches every combination of the configured interventions for the plan that keeps all designated critical
  services UP at the lowest cost, then with the fewest actions, then with the fewest residual failures;
- has the answer independently re-checked;
- explains why that plan won.

It **recommends**; a person approves. It executes nothing.

```
incident -> structured incident -> x(0) -> cascade (Φ) -> containment problem
        -> solver orchestrator (Branch & Bound, fallback Exhaustive, ... any registered solver)
        -> independent validator -> exact cross-check -> explanation -> human approval (simulated)
```

## Quick start (offline, standard library only)

Requires Python ≥ 3.10. Run all commands from the repository root.

```bash
python -m ripplecut demo                      # all canonical scenarios + a solver-failure resilience test
python -m ripplecut serve                     # local UI at http://127.0.0.1:8765/
python -m ripplecut run --text "paymentservice is down"
python -m ripplecut run --scenario payment_and_shipping_failure
python -m ripplecut run --failed paymentservice,shippingservice
python -m ripplecut run --fixture config/fixtures/synthetic_observed_checkout_down.json
python -m ripplecut run --scenario payment_failure --fault timeout   # resilience test: B&B forced to time out
python -m ripplecut run --scenario payment_failure --json            # full machine-readable run report
python -m ripplecut scenarios
python -m ripplecut validate-config
python -m ripplecut benchmark --out docs/BENCHMARK.md                 # measured Exhaustive vs B&B
```

Other fault modes for `--fault` are `crash`, `invalid_action`, `wrong_cost` and `malformed`. `--verbose` prints
the structured JSON log events to stderr.

## Tests

```bash
pip install pytest            # the only non-stdlib dependency, test-time only
python -m pytest              # 791 tests; ~9 s on the build machine
```

## Canonical scenarios

There is one topology, one rule set and one action space; only the incident changes. All scenarios are
**Controlled RippleCut Scenarios**. Every result below was produced by the planner, checked against a
hand-derived expectation, and validated.

| scenario | uncontrolled fixed point (DOWN) | recommended plan | (K, N, R) |
|---|---|---|---|
| payment_failure | checkout, payment | payment_fallback | (3, 1, 0) |
| shipping_failure | checkout, shipping | shipping_fallback | (2, 1, 0) |
| payment_and_shipping_failure | checkout, payment, shipping | checkout_backend_standby (beats both fallbacks (5, 2, 0) on N) | (5, 1, 0) |
| email_failure | email (soft edge: nothing propagates) | do nothing | (0, 0, 1) |
| productcatalog_failure | catalog, checkout, frontend, recommendation | restore_productcatalog_replica | (4, 1, 0) |
| currency_failure | currency, checkout, frontend | restore_currency_replica | (2, 1, 0) |
| cart_failure | cart, checkout, frontend | **NO_FEASIBLE_PLAN** (certified; no action restores cart) | — |
| observed_state_replay (SYNTHETIC fixture) | payment, checkout, recommendation (degraded) | payment_fallback + restart_checkoutservice | (4, 2, 1) |

## What is source-backed and what is modeled

- **SOURCE-BACKED TOPOLOGY.** The 11 services and every call edge, with file/line evidence from Online
  Boutique commit `38e7348` (`docs/TOPOLOGY_VERIFICATION.md`).
- **RIPPLECUT-MODELED.** The Boolean rules and the hard/soft classification (informed by the source's error
  handling), the criticality set, and every intervention and its cost. The interventions are hypothetical and
  do not exist in Online Boutique.
- **RCAEval.** Real inspected cases are integrated in `config/rcaeval_cases/` with exact time-series schema
  (`inject_time.txt`, `metrics.json`), explicit `SignalStatus` handling (`AVAILABLE`, `MISSING`, `AMBIGUOUS`),
  and strict architectural separation between RCA root-cause metrics and containment planning.
- **REFERENCE ORACLE.** Differential verification is proven by an independent, transparent reference oracle
  in `tests/reference_oracle.py` across all dependency rules, interventions, and lexmin criteria.

## Documentation

| file | contents |
|---|---|
| `docs/IMPLEMENTATION_REPORT.md` | the 25-section implementation report (master spec §109) |
| `docs/ARCHITECTURE.md` | pipeline, Solver Independence Invariant, trust boundary, package map |
| `docs/MATHEMATICS.md` | definitions as implemented; proofs of termination, monotonicity, B&B exactness, certificate soundness; the unsafe bound |
| `docs/TOPOLOGY_VERIFICATION.md` | services, edges, evidence, modeled rules, interventions |
| `docs/VALIDATION_REPORT.md` | gates 1–12, validator evidence, test inventory |
| `docs/BENCHMARK.md` | measured Exhaustive vs B&B results |
| `docs/JUDGE_QA.md` | short answers to the judges' questions |
| `docs/DOCUMENT_CLASSIFICATION.md` | authority of every source document; declared differences |

## Optional LLM

`--llm` uses Claude through the Messages API if `ANTHROPIC_API_KEY` is set. The model can be overridden with
`RIPPLECUT_LLM_MODEL`. The LLM can only propose a structured incident, which is strictly schema-validated, or
rephrase computed facts, which are checked for invented services, actions, costs and claims. Without a key, or
on any API error, the deterministic parser is used. The demo, the UI and the tests never need the network.

## Scope

RippleCut is a hackathon MVP. It is not production-ready, and it does not remediate anything automatically.
Its guarantees apply to the explicit finite model only.
