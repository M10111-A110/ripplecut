# Document Classification

Every supplied or consulted document, classified per master spec §1 and §129A. No documents were blended; where
two sources differ, the difference is declared below rather than reconciled silently.

| document | classification | how it was used |
|---|---|---|
| `RIPPLECUT_MASTER_IMPLEMENTATION_GREENFIELD_v2.txt` (supplied) | AUTHORITATIVE — IMPLEMENTATION | Build architecture: solver interface, registry, orchestrator, execution guard, validator, fallback, B&B, exhaustive oracle, gates, tests, offline demo, report format. Read in full. |
| `RippleCut_Technical_Understanding_FINAL_AUDITED.pdf` (supplied) | AUTHORITATIVE — TECHNICAL | Formal model: Boolean states, criticality and `C_min`, AND/OR/threshold rules, Φ and its fixed point, termination under `Φ(x) ⪯ x`, `T_B`, `x*_B = Cascade(T_B(x(0)))`, lexicographic `(K, N, R)` over `F_feas`, infeasibility reporting, LLM role, scope limits. |
| GoogleCloudPlatform/microservices-demo, commit `38e7348eb289eb5b87c0c6e8cb19ced0449dc389` (cloned during the build) | SOURCE-BACKED EXTERNAL | Service names and every call edge, with file/line evidence (`docs/TOPOLOGY_VERIFICATION.md`). Also the source of the hard/soft classification: errors that abort a request versus errors that are only logged. |
| RCAEval repository, commit `259ea41` (cloned during the build) | EVIDENCE — README/code only | Case-directory layout, as documented in the README (for example `re1ob_adservice_cpu_1`, containing `metrics.json` or, in the Hugging Face copy, `metrics.parquet`, plus `inject_time.txt`); RE2-OB is listed with 90 Online Boutique cases. **No dataset case was inspected**: the dataset hosts (Hugging Face / Figshare / Zenodo) were not reachable from the build sandbox. |
| RCAEval benchmark cases | BENCHMARK — **none available** | Nothing in RippleCut is labeled benchmark-backed. The adapter (`RCAEvalCaseSource`) raises `DATA_ADAPTER_ERROR` because its metric mapping is `UNKNOWN — REQUIRES VERIFICATION`. |
| `Team_3_idiots.pptx`, `Details-of-PS.txt`, `PPT_Round_Rubric.txt` | REFERENCE ONLY — **not supplied** | Not used. No presentation or rubric wording was assumed. |
| `config/fixtures/synthetic_observed_checkout_down.json` (written in this build) | SYNTHETIC (RippleCut-authored) | Offline replay input for the state estimator; its provenance label says so explicitly. It is not evidence. |

## Declared differences between authoritative sources

1. **Primary planner.** PDF §7.3 names exhaustive enumeration as the MVP planner. Master spec §2 deliberately
   extends this: Branch & Bound is the primary exact solver, and Exhaustive is retained as the oracle. By the
   priority order (master spec first), B&B is primary.

   The original model is not weakened. `docs/MATHEMATICS.md` Theorem 2 proves that a completed B&B returns
   exactly the oracle's `B*`, and every run in the default policy is cross-checked by the exhaustive oracle
   (`m ≤ 12`). The optimality claim therefore still rests on the finite enumeration semantics of PDF §7.3.

2. **`F_feas` and legality.** PDF §8.3 writes `F_feas = {B ⊆ A : C(B) ≥ C_min}`, and PDF §7.2 requires that
   conflicting combinations be marked invalid or given an explicit joint effect. RippleCut makes this explicit:
   `F_feas = {B legal : C(B) ≥ C_min}`. An invalid combination has no `T_B` and cannot be feasible.

3. **Forced failures and Φ.** PDF §5.3 speaks of retaining "externally forced failures", and §5.5 assumes
   `Φ(x) ⪯ x`. RippleCut implements `Φ(x)_v = x_v ∧ R_v(x)`, which treats every initially DOWN service as forced.
   `docs/MATHEMATICS.md` Proposition A shows that this gives the identical trajectory for every incident stated
   as a failure set.

4. **Final tie-breaker.** The PDF stops at `(K, N, R)`. Master spec §45 adds sorted action ids as a final
   deterministic tie-breaker. It is used only when `(K, N, R)` are all equal.
