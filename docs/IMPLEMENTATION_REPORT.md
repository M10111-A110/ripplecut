# RippleCut — Final Implementation Report

Format: exactly the 25 sections required by master spec §109. Every number is from a run made during the build
(`docs/VALIDATION_REPORT.md` and `docs/BENCHMARK.md` are generated, not typed). Nothing is claimed as
production-ready or benchmark-backed.

## 1. Implementation Summary

RippleCut was built greenfield in Python, using the standard library only at runtime. It has five parts:

- **Deterministic core.** Model loader and validation, a single dependency engine, the cascade simulator
  `Φ(x)_v = x_v ∧ R_v(x)`, an intervention engine with explicit composition, and a lexicographic objective
  `(K, N, R)` with a sorted-id tie-break.
- **Solver framework.** Interface, registry, config-driven policy, execution guard and orchestrator, with fallback
  and a verification layer. Branch & Bound is the primary exact solver, and Exhaustive is the oracle.
- **Independent safety validator.** It recomputes every result, with a monotone infeasibility certificate.
- **Input and output layers.** Structured incidents, a deterministic NL parser, an optional schema-guarded LLM
  parser, a state estimator with a SYNTHETIC replay fixture, an RCAEval adapter boundary, deterministic
  explanations, simulated human approval, a CLI, and an offline local UI.
- **Scenarios.** Eight canonical scenarios on one Online Boutique topology.

Status: **756 tests pass**, and all 12 implementation gates pass. All 8 scenarios match their hand-derived
expectations. B&B matched the exhaustive oracle in every comparison: 8 canonical scenarios and 170 seeded random
instances in the benchmark, plus 400 random seeds × 4 B&B variants in the tests.

## 2. Repository Changes

This is a new repository; no prior code existed. The layout is:

```
config/    canonical model (11 services, 9 hard + 6 soft edges, 8 actions, 3 conflicts), solver policy,
           8 scenarios, SYNTHETIC replay fixture
ripplecut/ engine, model, solvers, validation, incident, evidence, explain, pipeline, cli, benchmark,
           generators, ui
tests/     14 test modules, 756 tests
docs/      architecture, mathematics, topology verification, validation report, benchmark, judge Q&A,
           document classification, this report
scripts/   generators for TOPOLOGY_VERIFICATION.md and VALIDATION_REPORT.md
```

## 3. Architecture Implemented

The pipeline is: incident → structured incident → `x(0)` → cascade → ContainmentProblem → orchestrator. The
orchestrator applies the policy order and, for each solver, runs the guard and then the validator, falling back
on failure. An accepted result goes through the exact cross-check, then to the explanation, the UI/CLI, and
simulated approval.

The Solver Independence Invariant holds by construction: solvers see only the frozen problem and a
`SolverContext` that routes into the single engine. It is also checked statically by
`tests/test_architecture_integrity.py`. Details are in `docs/ARCHITECTURE.md`.

## 4. System Model

The 11 frozen services are listed in PDF §2.1. `loadgenerator` is infrastructure. redis-cart and
shoppingassistantservice are excluded (documented).

- **Critical (RippleCut policy):** frontend, checkoutservice, paymentservice, cartservice, productcatalogservice,
  currencyservice, shippingservice. `C_min = 7`.
- **State meaning:** `x = 1` iff UP. DEGRADED maps to 0. `x_frontend` models the browse path (home page), and
  `x_checkout` models whether PlaceOrder can complete. Both definitions are RIPPLECUT-MODELED.
- **Call edges:** every edge is SOURCE-BACKED with file/line evidence at Online Boutique commit `38e7348`
  (`docs/TOPOLOGY_VERIFICATION.md`).

## 5. Dependency Rules

AND, OR, THRESHOLD and GROUP (a conjunction of thresholds) are implemented once, in
`engine/dependency.py`, and are all proven monotone. The canonical model has four rules:

- `checkout = cart ∧ catalog ∧ currency ∧ shipping ∧ payment`, with email soft: a failed confirmation email is
  only logged.
- `frontend = catalog ∧ currency ∧ cart`, with ad, recommendation, shipping and checkout soft for the browse path.
- `recommendation = catalog`.
- `loadgenerator`: soft only, so it is a root.

Every rule is labeled RIPPLECUT-MODELED; no 2-of-3 rule is claimed for Online Boutique. The OR, THRESHOLD and
GROUP rule types are exercised on synthetic models.

## 6. Cascade Simulator

`engine/simulator.py` implements a synchronous `Φ` that runs to a fixed point. It returns the full round
history, the newly failed services per round, the number of state-changing rounds, the number of `Φ`
evaluations and the termination reason.

- `Φ(x) ⪯ x` and monotonicity hold by construction, for cyclic graphs as well.
- Termination takes at most `n` state-changing rounds (proof in `docs/MATHEMATICS.md`).
- A hard limit of `n + 1` evaluations yields an explicit `SIMULATION_ERROR`; it is unreachable for valid models.
- Proposition A shows the trajectory is identical to the PDF's forced-failure iteration for failure-set
  incidents.

## 7. Intervention Engine

`engine/interventions.py` checks legality before any simulation: unknown or duplicate ids, preconditions on
`x(0)`, INVALID_COMBINATION pairs, and an action taking part in two joint effects. `DEFINED_JOINT_EFFECT`
replaces the effects of its pair. `T_B` sets the services in `Up(B)` to 1 and those in `Down(B)` to 0.

Model validation rejects undeclared opposite effects. This makes `T_B` well defined (Lemma 4) and every
illegality upward closed (Lemma 5), which B&B relies on. The optimistic state (Lemma 6) supplies the bounds and
the certificate.

## 8. Objective Function

`engine/objective.py` computes `C, K, N, R` exactly as in PDF §8.1. Costs are exact `Fraction`s.
`F_feas = {B legal : C(B) ≥ C_min}`. The key is `(K, N, R, sorted ids)`, and it is the only order any solver uses.

There is no weighted score anywhere; a test demonstrates the "weighted-score trap". The hand-calculated §115
example is tested.

## 9. Exhaustive Solver

`solvers/exhaustive.py` evaluates all `2^m` subsets through the context and does no pruning. It reports the
top-5 feasible plans in its metadata and declares `max_actions = 16`. It is used as:

- the configured fallback;
- the exact cross-check (for `m ≤ 12`);
- the oracle in the tests and the benchmark.

## 10. Branch & Bound Solver

`solvers/branch_and_bound.py` searches a set-enumeration tree over the precondition-applicable actions, ordered
by (cost, id). It uses only safe rules:

- **S0:** precondition filter.
- **S1:** upward-closed illegality.
- **S2:** `(K + min remaining cost, N + 1)` lexicographic bound.
- **S3:** feasibility upper bound from the optimistic cascade.
- **S4:** residual lower bound.

Equality never prunes. There are two strategies: best-first, with a proven early stop, and depth-first.
Exactness is Theorem 2 in `docs/MATHEMATICS.md`. The tempting count bound `LB_N` is proven unsafe by a
counterexample test and is not used.

## 11. Solver Plugin Architecture

- `Solver` ABC with `name`, `capabilities` (EXACT or HEURISTIC, `max_actions`), `supports()` and `solve()`.
- `SolverRegistry` holds explicit instances and has no global state.
- `SolverPolicy` is read from `config/solver_policy.json`.
- `solvers/builtin.py` is the only module that names concrete solvers.

Tests register 2, then 3 trivial solvers, and a chain of 30, without touching the pipeline, orchestrator,
validator or UI.

## 12. Solver Fallback Mechanism

The guard runs each solver in a thread with a deadline and an evaluation budget. It maps failures to
`INCOMPATIBLE`, `ERROR`, `TIMEOUT`, `RESOURCE_LIMIT` or `INVALID_RESULT`; a solver that ignores cancellation is
abandoned and its result discarded.

The orchestrator validates each result and falls back in policy order. If nothing is accepted it returns
`NO_VALID_SOLVER_RESULT`. A validated incumbent from a timed-out solver is used only if
`accept_validated_incumbent_if_all_fail` is true (default: false).

The fault-injection resilience test covers five modes: crash, timeout, invalid_action, wrong_cost and malformed.
It is tested on 4 scenarios × 5 modes, and each fallback reproduces the normal result.

## 13. Independent Validator

`validation/validator.py` runs checks 1–10 from the problem definition and compares every reported value,
returning `INVALID_RESULT` on any mismatch. Infeasibility is handled in two tiers:

- a monotone certificate (sound, not complete);
- otherwise, acceptance only from a completed exact search, labeled uncertified.

In addition, the exact cross-check can upgrade HEURISTIC to PROVEN_OPTIMAL on an identical `(K, N, R)`, can
downgrade PROVEN_OPTIMAL on a mismatch, and constructively refutes a false infeasibility claim. Tests corrupt
cost, count, residuals, the objective tuple and the final state, and submit unknown, illegal, conflicting,
duplicate and infeasible plans; all are rejected.

## 14. RCAEval Integration

**Not benchmark-backed.** The RCAEval repository was cloned and its README read (commit `259ea41`: case layout,
and RE2-OB listed with 90 cases). The dataset hosts were unreachable from the build sandbox, so no case was
inspected.

- `RCAEvalCaseSource` checks the directory layout and then raises `DATA_ADAPTER_ERROR`, because the metric
  mapping is `UNKNOWN — REQUIRES VERIFICATION`.
- The observed-state path works through `ReplayFixtureSource` and the configured `StateEstimator`, using a
  fixture labeled SYNTHETIC.
- No scenario is attributed to RCAEval, and scenario loading rejects an unqualified "RCAEval" label.

## 15. LLM Integration

The LLM is optional and off by default. `incident/llm.py` is a stdlib Messages-API client that reads
`ANTHROPIC_API_KEY` and never logs it. The LLM parser accepts a JSON object whose keys come from a fixed allowed
set, whose service ids exist in the model, and which passes the same root-vs-symptom ambiguity check as the
deterministic parser.

Hallucinated actions, extra keys, malformed replies, unknown services and ambiguity are rejected; this is tested
with a stub client. If the LLM is unavailable, the deterministic parser takes over. LLM explanations must pass
`check_explanation`: no unknown services or actions, costs must match the facts, and no "automatically fixed"
claims.

**The real API was not called in this build** (no key in the sandbox). Only the stubbed contract is tested.

## 16. UI/Demo

`python -m ripplecut serve` runs a stdlib HTTP server with a single offline HTML page (no CDN). The page has six
numbered steps: Incident, System, Cascade, Planner, Result and Why. They include:

- an SVG graph with snapshots for `x(0)`, each cascade round, the fixed point, and the state with the plan;
- ambiguity alternatives shown as choices;
- the resilience-test toggle;
- validator checks;
- **Approve plan (simulated)**, which records `APPROVED_SIMULATED` with `executed: false`.

The page was screenshot-checked at desktop and mobile widths with headless Chromium. `python -m ripplecut demo`
runs everything in the terminal. The demo works with sockets blocked, as tested.

## 17. Test Coverage

756 tests in 14 modules (`docs/VALIDATION_REPORT.md`) cover:

- master spec §62: rules, cascade, interventions, objective, solvers, validator, LLM, reproducibility;
- master spec §63 property tests: B&B equivalence, determinism, validator consistency, termination,
  monotonicity, upward closure, optimistic dominance;
- all 20 edge cases of §64;
- the §114–§122 tests;
- the canonical scenarios end-to-end, including natural language, the UI API contract, structured logging, and
  the offline demo.

Line coverage was not measured; `coverage.py` is not installed.

## 18. B&B vs Exhaustive Results

These are measured values (`docs/BENCHMARK.md`, Python 3.12.3). On the canonical scenarios (m = 8, 256
candidates), B&B evaluated 1–7 plans in 0.12–0.39 ms, against 1.40–1.67 ms for Exhaustive, with identical plans
and objectives.

On seeded random instances (14 services, cycles, all rule types, conflicts, joint effects; m = 8…16), 170 of 170
were equivalent. Median Exhaustive time rose from ≈2.5 ms at m = 8 to ≈0.5 s at m = 16. Median B&B time stayed
below 4 ms, with at most 298 nodes.

For m = 20 and 24, B&B only was run: at most 776 nodes, and the depth-first and best-first strategies agreed.
No equivalence is claimed there. The instances are easy for B&B because optimal plans are small; the worst case
remains exponential.

## 19. Known Limitations

- **Model fidelity.** The rules, criticality, interventions and costs are RippleCut-modeled; only the call edges
  are source-backed. The Boolean MVP has no load, retry, queue, timing or recovery dynamics.
- **Validator scope.** The validator shares the single engine by design. It detects solver faults, not engine
  bugs.
- **Parser and certificate.** Both are intentionally incomplete (they refuse rather than guess). NL phrasing
  such as "checkout orders failing" alongside "payment down" is returned as AMBIGUOUS with two explicit options.
- **Guard.** Python threads cannot be killed; an uncooperative solver is abandoned but keeps consuming CPU
  until it finishes.
- **Cross-check.** The exact cross-check runs only for `m ≤ 12`.
- **Packaging.** Configuration is read from the repository's `config/` directory; run from the repo root.

## 20. Remaining Risks

- **Presentation.** Judges may read modeled rules as Online Boutique facts; every label says otherwise, but the
  presenter must repeat it.
- **Real LLM path.** It is untested against the live API.
- **RCAEval.** Case integration awaits access to real data; the adapter will need its metric mapping written
  after inspecting a case.
- **B&B performance.** It is characterized only on small or random models. Adversarial instances can approach
  `2^m`.
- **UI.** It was visually checked in Chromium only.

## 21. How to Run

```bash
cd ripplecut                                  # repository root (Python >= 3.10, no dependencies)
python -m ripplecut validate-config
python -m ripplecut run --text "paymentservice is down"
python -m ripplecut run --scenario payment_and_shipping_failure
python -m ripplecut run --failed paymentservice,shippingservice --json
python -m ripplecut run --fixture config/fixtures/synthetic_observed_checkout_down.json
python -m ripplecut benchmark --out docs/BENCHMARK.md
```

## 22. How to Run Tests

```bash
pip install pytest
python -m pytest                              # 756 passed in ~8 s on the build machine
python scripts/gen_validation_report.py       # regenerates docs/VALIDATION_REPORT.md from a real run
```

## 23. How to Run Demo

```bash
python -m ripplecut demo                                   # 8 scenarios + resilience test, offline
python -m ripplecut run --scenario payment_failure --fault timeout
python -m ripplecut serve                                  # UI at http://127.0.0.1:8765/
```

Suggested live flow (master spec §106):

1. Type "paymentservice is down" and run the planner.
2. Step through the cascade snapshots.
3. Show that B&B's result is validated as PROVEN_OPTIMAL.
4. Read the Why panel.
5. Approve (simulated).
6. Pick "paymentservice + shippingservice", then "cartservice" (no feasible plan).
7. Enable the resilience test and rerun.

## 24. Files Changed

Every file is new (greenfield):

- `README.md`, `pyproject.toml`, `.gitignore`
- `config/online_boutique_canonical.json`, `config/solver_policy.json`, `config/scenarios.json`,
  `config/fixtures/synthetic_observed_checkout_down.json`
- `ripplecut/__init__.py`, `__main__.py`, `errors.py`, `logs.py`, `pipeline.py`, `cli.py`, `benchmark.py`,
  `generators.py`
- `ripplecut/model/{schema,loader,validation}.py`
- `ripplecut/engine/{dependency,simulator,interventions,objective}.py`
- `ripplecut/solvers/{base,registry,policy,guard,orchestrator,builtin,fault_injection,branch_and_bound,exhaustive}.py`
- `ripplecut/validation/validator.py`
- `ripplecut/incident/{schema,parser,llm}.py`
- `ripplecut/evidence/{state_estimator,sources}.py`
- `ripplecut/explain/{explainer,llm_explainer}.py`
- `ripplecut/ui/server.py`, `ripplecut/ui/static/index.html`
- `tests/` (conftest plus 14 modules)
- `docs/` (8 documents)
- `scripts/gen_topology_doc.py`, `scripts/gen_validation_report.py`

## 25. Recommended Next Steps

1. Inspect one RE2-OB Online Boutique case, then write and test the metric mapping in `RCAEvalCaseSource`.
   Label it RCAEVAL_INSPECTED only after that.
2. Exercise the live LLM path with a key and record its parse accuracy on the tested phrasings.
3. Add a heuristic solver (for example greedy) behind the existing interface, to show the HEURISTIC →
   cross-check path live.
4. Measure line coverage and fill any gaps.
5. Consider multiprocessing isolation in the guard, if hard kills of uncooperative solvers are needed.
6. Only with evidence: refine rules for other Online Boutique pages (for example a cart-page variable) as new
   modeled variables, never as claimed production semantics.
