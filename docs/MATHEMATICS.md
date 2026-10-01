# RippleCut — Mathematics as Implemented

This document states the definitions that the code actually implements and proves the properties the code
relies on. All guarantees hold **for the explicit finite RippleCut model only** (PDF §1.4, §9.4). They say
nothing about how real Online Boutique behaves.

Code references are given as `module:function`.

---

## 1. State space and rules

Let `V` be the service set, `n = |V|`, and `X = {0,1}^V` with `x_v = 1` iff `v` is UP (PDF §4.1). DEGRADED and
DOWN both map to `0`. Order `X` componentwise: `x ⪯ y ⟺ x_v ≤ y_v for all v`.

Each service `v` has either no rule (then `R_v ≡ 1`, a root) or one rule over its **hard** inputs
`D_v = {u_1, …, u_k}` (`engine/dependency.py:CompiledRule.evaluate`):

| type | definition |
|---|---|
| AND | `R_v(x) = ∏_i x_{u_i}` (empty product = 1) |
| OR | `R_v(x) = 1[Σ_i x_{u_i} ≥ 1]` |
| THRESHOLD | `R_v(x) = 1[Σ_i x_{u_i} ≥ q]`, `1 ≤ q ≤ k` |
| GROUP | `R_v(x) = ∏_g 1[Σ_{u∈g} x_u ≥ q_g]` (conjunction of thresholds over a partition-cover of `D_v`) |

**Soft** inputs are recorded topology and are never read by `R_v` (a soft upstream failure never propagates).
Parameter ranges are enforced by `model/validation.py:model_issues`.

**Lemma 1 (monotone rules).** Every `R_v` is monotone: `x ⪯ y ⇒ R_v(x) ≤ R_v(y)`.

*Proof.* `x ⪯ y` implies `Σ_{u∈S} x_u ≤ Σ_{u∈S} y_u` for every `S`, and `t ↦ 1[t ≥ q]` is nondecreasing, so every
threshold indicator is monotone. AND is the threshold `q = k`; OR is `q = 1`; a product (conjunction) of
monotone `{0,1}`-valued functions is monotone. ∎ (Checked exhaustively on all `2^7` states in
`tests/test_dependency_rules.py::test_every_rule_type_is_monotone`.)

## 2. The cascade map Φ

Implemented transition (`engine/simulator.py:phi`):

```
Φ(x)_v = x_v ∧ R_v(x)
```

**Lemma 2 (descent).** `Φ(x) ⪯ x` for every `x`. *Proof.* `x_v ∧ R_v(x) ≤ x_v`. ∎

**Lemma 3 (monotonicity).** `x ⪯ y ⇒ Φ(x) ⪯ Φ(y)`. *Proof.* `x_v ≤ y_v`, `R_v(x) ≤ R_v(y)` (Lemma 1), and `∧` is
monotone in both arguments. ∎

Both lemmas hold **by construction for every configured model, including cyclic dependency graphs**; no
acyclicity is assumed (master spec §65).

**Theorem 1 (termination).** From any `x(0)`, the iteration `x(t+1) = Φ(x(t))` reaches a fixed point
`x* = Φ(x*)` after at most `|x(0)|₁ ≤ n` state-changing rounds, i.e. at most `n + 1` evaluations of `Φ`.

*Proof.* By Lemma 2 the sequence is ⪯-descending. A state-changing round changes at least one coordinate, and
by descent it can only change `1 → 0`, so `|x(t)|₁` strictly decreases in each such round. It is bounded below
by 0, so at most `|x(0)|₁` state-changing rounds occur; the next evaluation returns the same state. ∎

This is PDF Proposition 1 (§5.5) with its hypothesis `Φ(x) ⪯ x` made true by construction. The simulator
nevertheless enforces a hard limit of `n + 1` evaluations and returns `ROUND_LIMIT_EXCEEDED` /
`SIMULATION_ERROR` if it is ever reached, and re-checks descent defensively
(`NO_RECOVERY_ASSUMPTION_VIOLATED`). Neither branch is reachable for a validated model; both exist so that a
future rule type that broke the assumptions would produce an explicit diagnostic instead of a silent claim.

**Proposition A (agreement with the PDF's forced-failure iteration).** PDF §5.3 describes rounds that "retain
externally forced failures" and evaluate the rules: with forced set `F`,
`y(t+1)_v = 0` for `v ∈ F` and `y(t+1)_v = R_v(y(t))` otherwise. If `x(0) = y(0)` has `x(0)_v = 1` for every
`v ∉ F` and `0` on `F`, then `x(t) = y(t)` for all `t`.

*Proof.* First, `y` is descending: `y(1) ⪯ y(0)` because `y(0)` is 1 outside `F`; if `y(t) ⪯ y(t−1)` then
`R(y(t)) ⪯ R(y(t−1))` (Lemma 1) and so `y(t+1) ⪯ y(t)`. Now induct on `t`. On `F` both sequences are 0. For
`v ∉ F` we need `R_v(y(t)) = y(t)_v ∧ R_v(y(t))`, i.e. `R_v(y(t)) ≤ y(t)_v`: for `t = 0`, `y(0)_v = 1`; for
`t ≥ 1`, `y(t)_v = R_v(y(t−1)) ≥ R_v(y(t))` by descent and Lemma 1. ∎

So for every incident given as a failure set (all canonical structured scenarios), RippleCut's `Φ` produces
exactly the PDF's trajectory. The two formulations differ only when `x(0)` contains a service that is DOWN,
is not declared forced, and has all inputs UP (for example an observed-DOWN service in the replay fixture).
RippleCut treats **every initially DOWN service as forced (latched)**. This is precisely the PDF's
no-spontaneous-recovery hypothesis; it is also why the replay scenario needs `restart_checkoutservice`.

**Cascade operator.** `Cascade(x) := Φ^n(x)` (the fixed point is reached within `n` rounds and is then
stable). As a composition of monotone maps, `Cascade` is monotone: `x ⪯ y ⇒ Cascade(x) ⪯ Cascade(y)`
(property-tested on random models in `tests/test_simulator.py::test_termination_descent_and_monotonicity_random`).

**Cost of a simulation.** One `Φ` evaluation is `O(Σ_v |D_v|)`; a cascade is at most `n + 1` evaluations.

## 3. Interventions

`A = {a_1, …, a_m}` is finite (PDF §7.1). Each action has a cost `k(a) ≥ 0` (an exact rational), preconditions
evaluated on the **pre-intervention** state `x(0)`, and an effect `(set_up(a), set_down(a))`. Pairwise
conflicts are declared as `INVALID_COMBINATION` or `DEFINED_JOINT_EFFECT` with an explicit joint effect.

**Legality** (`engine/interventions.py:check_plan`), checked before any simulation. `B ⊆ A` is illegal iff one
of the following holds:

- (L1) `B` contains an unknown or duplicated id;
- (L2) some `a ∈ B` has a precondition that is false on `x(0)`;
- (L3) some `INVALID_COMBINATION` pair is contained in `B`;
- (L4) some action belongs to two `DEFINED_JOINT_EFFECT` pairs that are both contained in `B`.

**Effect units.** Each active joint pair contributes its joint effect *instead of* its two members' effects;
every other action contributes its own effect. `Up(B)`/`Down(B)` are the unions of the units' `set_up`/`set_down`.

```
T_B(x)_v = 1 if v ∈ Up(B);   0 if v ∈ Down(B);   x_v otherwise.
x*_B = Cascade(T_B(x(0)))                                    (PDF §7.2)
```

**Lemma 4 (well-definedness).** For a model accepted by `model_issues`, every legal `B` has
`Up(B) ∩ Down(B) = ∅`, so `T_B` is a well-defined deterministic map.

*Proof.* Each unit is internally consistent (validated). Suppose units `U₁ ≠ U₂` clash. (i) Two individual
actions `a, b` with opposite effects: validation requires a declared conflict on `{a,b}`; if it is INVALID, `B`
is illegal (L3); if it is a joint effect, `{a,b}` is an active joint pair, so neither is an individual unit —
contradiction. (ii) The joint effect of pair `P` and an individual action `c ∉ P`: validation requires a declared
conflict between `c` and some `x ∈ P`; INVALID gives L3; a joint effect makes `x` a member of two active joint
pairs, L4. (iii) Two joint effects of pairs `P₁, P₂`: if they share an action, L4; otherwise validation again
requires a declared conflict between members, which yields L3 or L4. ∎ (The code still re-checks
`Up ∩ Down = ∅` and reports it instead of resolving it silently.)

**Lemma 5 (illegality is upward closed).** If `B` is illegal then every `B' ⊇ B` is illegal.

*Proof.* Each of L1–L4 is witnessed by a subset of `B` (an id, a failing action, a pair, two pairs), which is
still contained in `B'`. ∎ (Exhaustively checked on random models in
`tests/test_interventions.py::test_illegality_is_upward_closed_random`.)

**Lemma 6 (optimistic state).** For `I ⊆ A` and candidates `Q`, let `P = I ∪ {q ∈ Q : pre(q) holds on x(0)}` and
let `U(P)` be the union of `set_up(a)` over `a ∈ P` together with the joint `set_up` of every joint pair
contained in `P`. Define `x_opt = x(0)` with every `v ∈ U(P)` set to 1. Then every **legal** `D` with
`I ⊆ D ⊆ I ∪ Q` satisfies `T_D(x(0)) ⪯ x_opt`
(`engine/interventions.py:optimistic_state`).

*Proof.* A legal `D` contains no precondition-failing action, so `D ⊆ P`, and every unit of `D` has its `set_up`
inside `U(P)`; hence `Up(D) ⊆ U(P)`. For `v ∈ Up(D)` both sides are 1; for `v ∈ Down(D)` the left side is 0;
otherwise the left side is `x(0)_v ≤ x_opt,v`. ∎ (Checked by enumeration in
`tests/test_interventions.py::test_optimistic_state_dominates_every_legal_completion`.)

**Corollary 1.** With `C(x) = Σ_v c(v) x_v` (monotone) and `R(x) = Σ_v 1[x_v = 0]` (antitone), monotonicity of
`Cascade` gives, for every such `D`: `C(D) ≤ C(Cascade(x_opt)) =: UB_C` and `R(D) ≥ R(Cascade(x_opt)) =: LB_R`.

## 4. Objective (PDF §8)

For a legal plan `B` with `x* = x*_B` (`engine/objective.py:compute_objective`):

```
C(B) = Σ_v c(v) x*_v        K(B) = Σ_{a∈B} k(a)        N(B) = |B|        R(B) = Σ_v 1[x*_v = 0]
C_min = Σ_v c(v)            F_feas = { B ⊆ A : B legal, C(B) ≥ C_min }
B* = lexmin_{B ∈ F_feas} (K(B), N(B), R(B)), then the sorted tuple of action ids.
```

The only difference from PDF §8.3 is the explicit "B legal" in `F_feas`: PDF §7.2 requires that combinations
without a defined composition are marked invalid, and an invalid combination has no `T_B`. The final
sorted-ids tie-breaker (master spec §45) makes the order total, so `B*` is unique and reproducible; it is only
consulted when `(K, N, R)` are all equal. No weighted score exists in the decision path
(`tests/test_objective.py::test_no_weighted_score_trap`). Costs are `fractions.Fraction`, so ties are never
broken by floating-point rounding. PDF §9.2 (existence of a minimum whenever `F_feas ≠ ∅`) applies unchanged.

**Proposition B (canonical model: `suppress_emailservice` is never selected).** Legality is downward closed
(contrapositive of Lemma 5), so for a legal `B ∋ suppress` the plan `B' = B ∖ {suppress}` is legal. If
`restart_emailservice ∉ B`, then `Up(B') = Up(B)` and `Down(B') ⊆ Down(B)`. If it is in `B`, the joint unit
(`emailservice` DOWN) is replaced by restart's own effect (`emailservice` UP). Either way `T_{B'}(x(0)) ⪰ T_B(x(0))`,
so by monotonicity `C(B') ≥ C(B)` and `R(B') ≤ R(B)`. Since `K(B') = K(B)` (its cost is 0) and `N(B') = N(B) − 1`,
`B'` is strictly better. Hence no plan containing the action is optimal. It is kept in the model to show that
the planner does not pick an action just because it is free. (This argument uses the specific joint effect; it is
not a general claim about every `set_down` action, because a joint effect could set UP something neither member
does alone.)

## 5. Exhaustive oracle

`solvers/exhaustive.py` evaluates all `2^m` subsets (PDF Proposition 2, §9.1) through the canonical evaluator
and keeps the key-minimal feasible one. Completing the enumeration therefore returns `B*` by definition.

## 6. Branch & Bound

Search space: the set-enumeration tree over the **applicable** actions `A₀ = {a : pre(a) holds on x(0)}`, sorted
by `(cost, id)` as `a_1, …, a_p`. A node is `B` with largest index `j`; its children are `B ∪ {a_k}`, `k > j`.
Every subset of `A₀` appears exactly once. The rules below discard a node's *strict descendants* only; the node
itself is always evaluated when visited.

| rule | discards | justification |
|---|---|---|
| S0 | every subset containing an action outside `A₀` | illegal by L2 |
| S1 | descendants of an illegal `B` | Lemma 5 |
| S2 | descendants if `(K(B) + k(a_{j+1}), N(B) + 1) >_lex (K*, N*)` | every descendant `D` has `K(D) ≥ K(B) + k(a_{j+1})` (costs sorted, nonnegative, `D` adds some `a_k`, `k > j`) and `N(D) ≥ N(B) + 1` |
| S3 | descendants if `UB_C < C_min` with `I = B`, `Q = {a_k : k > j}` | Corollary 1: no descendant is feasible |
| S4 | descendants if `(LB_K, LB_N, LB_R) >_lex (K*, N*, R*)` | Corollary 1 and the S2 bounds |

Here `(K*, N*, R*)` is the incumbent (best feasible key found so far).

**Lemma 7 (componentwise bounds imply lexicographic bounds).** If `D` satisfies `K(D) ≥ L_K`, `N(D) ≥ L_N`,
`R(D) ≥ L_R` and `(L_K, L_N, L_R) >_lex (K*, N*, R*)`, then `(K(D), N(D), R(D)) >_lex (K*, N*, R*)`.

*Proof.* Let `i` be the first coordinate where `L` exceeds the incumbent. The earlier coordinates of `D` are
`≥` the incumbent's, since they are `≥` equal values of `L`. If any of them is strictly greater, `D` is
lexicographically greater. Otherwise they are all equal, and `D`'s coordinate `i` is `≥ L_i >` the incumbent's.
∎ Equality never prunes: a subtree whose bound ties the incumbent may still win on `R` or on the id tie-break.

**Theorem 2 (exactness).** If the search completes, Branch & Bound returns exactly `B*`, the same plan as the
exhaustive oracle. It reports `NO_FEASIBLE_PLAN` iff `F_feas = ∅`.

*Proof.* The incumbent changes only to a strictly smaller key, so at every moment `key(incumbent) ≥ key(B*)`.
Suppose `B*` is never evaluated. Then some ancestor's subtree containing `B*` was discarded by S0–S4:

- S0 and S1 discard only illegal plans, but `B*` is legal.
- S3 discards only infeasible plans.
- S2 and S4 discard only plans strictly worse than the incumbent at that moment (Lemma 7), hence strictly worse
  than `key(B*)`. That contradicts the optimality of `B*`.

So `B*` is evaluated and becomes the incumbent, and nothing can replace it. The `NO_FEASIBLE_PLAN` case follows
the same way. ∎

**Best-first stopping rule.** The best-first variant pops nodes in `(K, N, ids)` order. A child has
`(K + k(a), N + 1) >_lex (K, N)`, so the popped `(K, N)` values are lexicographically nondecreasing. When a
popped node has `(K, N) >_lex (K*, N*)`, every node still in the heap, and every descendant of such a node, has
`(K, N)` at least as large. All of them are strictly worse, so stopping is safe. The depth-first variant has no
such rule and relies on S0–S4 only. Both variants are verified against the oracle.

**Not used: the naive count bound.** `LB_N = N(B) + ⌈(remaining critical failures) / (max critical services
restored by one action)⌉` is **unsafe**, because recoveries are super-additive under AND rules. Counterexample
(`tests/test_branch_and_bound.py::test_naive_count_bound_is_unsafe`):

- `X = A ∧ B`, with `X` critical and `A, B` DOWN.
- The actions `fa` and `fb` each restore one input.
- Each single action restores 0 critical services, but `{fa, fb}` restores 1.

The naive bound would prune the only feasible plan.

**Limits.** B&B is exact but not sub-exponential: its worst case is `2^m` nodes. A run interrupted by the time or
evaluation budget reports `TIMEOUT` / `RESOURCE_LIMIT` and is never labeled `PROVEN_OPTIMAL`.

## 7. Independent validation

For every proposed plan, `validation/validator.py:SafetyValidator.validate_plan` recomputes everything from
the problem definition, without using the solver's numbers. The checks are:

1. the plan exists;
2. every action id is known;
3. the combination is legal;
4. `T_B` is applied;
5. the cascade is re-run to a fixed point;
6. `C(B) ≥ C_min`;
7. `K` matches the reported value;
8. `N` matches;
9. `R` matches;
10. the reported `C`, objective tuple and final state all match the recomputation.

Any mismatch yields `INVALID_RESULT`.

**Infeasibility claims.** The validator's certificate is the root instance of Lemma 6 (`I = ∅`, `Q = A`): if
`C(Cascade(x_opt)) < C_min`, then Corollary 1 gives `C(B) < C_min` for every legal `B`, so `F_feas = ∅`. This
proof is **sound but not complete**. It cannot see conflicts, joint effects that replace their members, or
`set_down` effects, and `tests/test_validator.py::conflict_only_infeasible` is an infeasible instance it cannot
certify. Such a claim is accepted only from an EXACT solver whose search completed, only if the policy allows
it, and it is labeled "not independently certified".

If a configured exact cross-check solver then returns a validated feasible plan, that plan is a constructive
refutation. RippleCut replaces the claim with that plan and records `REFUTED_BY_CROSS_CHECK`.

**Scope of independence.** The validator is independent of every *solver*. It deliberately shares the single
engine (`check_plan`, `transform`, `simulate`, `compute_objective`), because the specification forbids duplicate
simulators and objective functions (master spec §97–§99). It therefore detects wrong, misreported or malicious
solver output, but not a bug in the engine itself. Engine correctness rests on:

- the hand-calculated examples (`tests/test_objective.py`, the §114 A/B/C test);
- the property tests of Lemmas 1–6;
- oracle equivalence.

## 8. Optimality labels (derived by RippleCut, never taken from the solver)

| label | when |
|---|---|
| `PROVEN_OPTIMAL` | an EXACT solver completed its search and its plan was validated; or a validated result was matched by a completed exact cross-check (same `(K, N, R)`) |
| `VALID_NOT_PROVEN_OPTIMAL` | a validated incumbent of an interrupted solver (only if policy allows), or a proven label downgraded by a cross-check mismatch |
| `HEURISTIC` | a validated result from a solver declared HEURISTIC |
| `NOT_APPLICABLE` | the final outcome is `NO_FEASIBLE_PLAN` (there is no plan to rank) |
| `UNKNOWN` | no valid solver result |

## 9. State estimation (PDF §3.2)

Each observation is `z_v = [latency_p95_ms, error_rate, request_rate, cpu_utilization]`. The descriptive state
uses thresholds from the model configuration:

- DOWN if `error_rate ≥ 0.5`;
- else DEGRADED if `error_rate ≥ 0.05`, or `latency ≥ 1000 ms`, or `cpu ≥ 0.9`;
- else UP.

The formal value is `x_v = 1[state = UP]`. These thresholds are configuration choices, not learned or
universal values, and `request_rate` is validated but not used. A missing observation is an error; no state
is ever assumed.

## 10. What is not proven

- That any Boolean rule, criticality choice, cost or intervention matches real Online Boutique behavior. The
  rules are RIPPLECUT-MODELED, and the interventions are hypothetical.
- Anything about load, retries, queues, recovery dynamics or timing (PDF §6.3, §17). None of these is modeled.
- Completeness of the natural-language parser or the infeasibility certificate. Both are sound-by-refusal:
  they return an explicit status rather than guess.
- Real-time bounds. A solver that ignores cancellation keeps running in an abandoned daemon thread (Python
  cannot kill threads). Its result is discarded.
