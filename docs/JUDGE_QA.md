# Judge Q&A (master spec §108 J)

**What is RippleCut?**
A containment planner for *modeled* cascading failures. Given an incident, it:

1. simulates how the failure spreads through an explicit service model of Online Boutique;
2. searches a finite set of configured interventions for the plan that keeps every designated critical service
   UP at the lowest cost, then with the fewest actions, then with the fewest residual failures;
3. has every solver answer independently re-checked;
4. explains the choice.

It recommends; a person approves. It executes nothing.

**Why not ChatGPT?**
A language model can describe a plausible fix, but it cannot guarantee that the fix satisfies the model's
constraints or that no cheaper feasible plan exists. In RippleCut the decision is computed by a deterministic
simulator and an exact search over all `2^m` candidate plans, and every result is re-verified. The LLM is
optional. It may only turn text into a schema-checked incident, or rephrase computed facts, and the rephrased
text is checked for invented services, actions, costs and claims. Without an API key everything still runs.

**Why not ServiceNow?**
RippleCut does not replace incident-management, observability or ITOM platforms, and we make no claim about what
those products can or cannot do. Its focus is narrow:

- an explicit dependency model;
- deterministic cascade simulation;
- a finite intervention search with a stated objective;
- replaceable optimization algorithms;
- independent result validation.

It could sit next to such tools as a decision aid.

**What is the mathematical model?**
Services have Boolean states (`x_v = 1` iff UP; DEGRADED counts as 0). Each service has a monotone rule over its
hard dependencies (AND, OR, threshold, or a conjunction of thresholds). An intervention plan `B` is a legal
subset of the action set and acts as a deterministic map `T_B`. The outcome is `x*_B = Cascade(T_B(x(0)))`. We
minimise `(K, N, R)` — cost, number of actions, residual failures — lexicographically over the plans that keep all
critical services UP, with sorted action ids as the final tie-breaker (`docs/MATHEMATICS.md`).

**What is Φ?**
The one-round cascade map, `Φ(x)_v = x_v ∧ R_v(x)`. A service stays UP only if it was UP and its rule is
satisfied. By construction `Φ(x) ⪯ x` (no spontaneous recovery) and `x ⪯ y ⇒ Φ(x) ⪯ Φ(y)` (monotonicity), for
any graph, cyclic or not.

**What is a fixed point?**
A state with `Φ(x*) = x*`: one more round changes nothing, so the cascade has finished. Every state-changing
round turns at least one 1 into a 0, so a fixed point is reached after at most `n` such rounds. The simulator
still enforces `n + 1` evaluations as a hard limit and returns an explicit error instead of looping.

**Why Branch & Bound?**
Exhaustive search evaluates all `2^m` plans. B&B returns the same optimum (Theorem 2 in `docs/MATHEMATICS.md`)
while skipping subtrees that are provably illegal, provably infeasible, or provably strictly worse. Its bounds
are sound because the cascade is monotone. We also document a tempting bound that is *unsafe* and therefore not
used.

Measured on the canonical scenarios (m = 7, 128 plans), B&B evaluated 1–4 plans; on seeded random instances it
matched the oracle in all 170 comparisons (`docs/BENCHMARK.md`). Its worst case is still exponential; we claim
exactness, not speed guarantees.

**Why multiple solvers?**
Solver choice is configuration, not architecture. A new algorithm (heuristic, ILP, SAT, …) plugs in behind the
same interface. The rest of the system — model, simulator, objective, validator, UI — does not change, and
because RippleCut derives the optimality label itself, a heuristic can never pass itself off as proven optimal.

**What happens if a solver fails?**
Every solver runs behind an execution guard that converts crashes, timeouts, budget exhaustion and malformed
output into explicit statuses. The orchestrator then tries the next solver in the configured fallback order. If
every solver fails or is rejected, the answer is `NO_VALID_SOLVER_RESULT` and no plan is fabricated. The demo has
a labeled "Solver failure simulation / resilience test" that breaks B&B in five different ways; Exhaustive takes
over, and its validated result is identical.

**How do you know the answer is valid?**
The validator never trusts solver numbers. It re-checks the action ids and legality, reapplies `T_B`, re-runs
the cascade, and recomputes `C, K, N, R` and the final state, rejecting any mismatch. Tests feed it deliberately
corrupted results: wrong cost, wrong count, wrong residuals, wrong state, unknown or illegal actions, a plan that
loses a critical service. All are rejected.

On top of that, the exhaustive oracle cross-checks the accepted answer (m ≤ 12). Infeasibility is proven by a
monotone certificate where possible and labeled "not independently certified" where not.

One honest limit: the validator shares the single engine by design, so it catches solver errors, not engine
bugs. The engine is covered by hand-calculated tests and property tests.

**What does RCAEval contribute?**
RCAEval provides the directory layout convention, case naming structure, and evaluation benchmark design.
In P0, RippleCut bundles three synthetic-derived test fixtures (`config/rcaeval_cases/`, e.g. `re1ob_cartservice_cpu_1`)
with explicit multi-metric time-series schema and injection timestamps. **The bundled data is synthetic telemetry,
not original RCAEval dataset data.** Only `re1ob_cartservice_cpu_1` corresponds to a real case ID in the official
dataset. RippleCut enforces a strict boundary:
1. Signal status: metrics are classified as `AVAILABLE`, `MISSING`, `UNSUPPORTED`, or `AMBIGUOUS`. Missing metrics are
   tracked explicitly and never silently converted into healthy defaults.
2. Separation of metrics: RCA root-cause detection metrics evaluate anomaly detection (whether the ground-truth
   service appears among anomalous services), while containment metrics (cost K, interventions N, residual
   failures R) evaluate safety interventions.
3. Provenance: Every bundle is tagged with exact provenance (`SYNTHETIC_DERIVED`, `SYNTHETIC_REPLAY`). Real RCAEval
   data ingestion is deferred to P1.

**What are the limitations?**

- The rules, criticality set, interventions and costs are RippleCut modeling choices. Only the call edges are
  source-backed.
- There is no load, retry, queue or recovery dynamics (Boolean MVP).
- Guarantees are about the model, not production.
- The Python guard cannot kill a solver that ignores cancellation; it abandons it and discards its result.
- B&B is exponential in the worst case.
- The natural-language parser and the infeasibility certificate are deliberately incomplete: they refuse
  rather than guess.
