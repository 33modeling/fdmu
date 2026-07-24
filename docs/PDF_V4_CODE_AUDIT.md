# PDF Chapter 4 ↔ code audit

Audit date: 2026-07-24
Normative specification: `KDD_UnlearningFail.pdf` (13 pages)
Audited repository revision: `98db4b2` plus the clean working tree present at audit start

Remediation status: the working tree now contains a tested PDF-v4 repair core
and several unambiguous fixes listed below. These changes do **not** by
themselves make the campaign claim-ready because the PDF's open constants and
the RQ1--RQ3 evidence/executor integration remain unresolved.

## Bottom line

The July 24 PDF is materially newer than the implementation, the checked-in
LaTeX, the preregistration constants, and the top-level README. The repository
is **not yet capable of producing a claim-valid run under the PDF contract**.

The numerical loss-shake estimator in Sections 4.1--4.2 is largely aligned.
The first-reaching parent path exists in the alpha-protection runner. At audit
start the main break began at Section 4.4: its repair objective and guard were
the old mean-loss/one-sided-drift design. A separate v4 core now implements the
new mechanics, but the active campaign still uses the legacy path. The
paper-facing statistics also implement an older two-claim contract and cannot
decide the new RQ1--RQ3 requirements.

This is an audit, not a claim that every older component is unusable. Legacy
gate and crossed-channel experiments remain useful diagnostics, but their
outputs must not be labeled as evidence for the current PDF.

## Chapter 4 compatibility matrix

| PDF section | Required behavior | Current status | Main evidence |
|---|---|---|---|
| 4.1 loss-shake | normalized shared Gaussian directions, symmetric response, `d_B/R` squared aggregation, signed responses saved, fp32 confirmatory path | **mostly aligned** | `src/rsus/probe/finite_diff.py:105-191`, `src/rsus/probe/fidelity.py:54-136` |
| 4.2 proximity and mixture | answer-token mean final hidden state, top-k forget cosine, discovery-fitted empirical midranks, separate `alpha_pred` and `alpha_prot` | **core functions aligned; end-to-end path incomplete** | `src/rsus/probe/baselines.py:92-164`, `src/rsus/analysis/mixture.py:43-135`, `:154-390` |
| 4.3 conditional prediction | damage at the first saved checkpoint satisfying the external direct-forgetting gate; non-reaching excluded from conditional outcomes | **one runner aligned, legacy runner violates it** | aligned: `experiments/channel_matrix/alpha_protection.py:1286-1322`; violation: `experiments/gate_1p5b/gate.py:696-715`, stale declaration in `src/rsus/generators/base.py:1-8` |
| 4.4 Eq. (7) repair | `L_P + beta KL(p_entry || p_theta)` on a selector-independent neutral stream | **implemented in new core; legacy runner not migrated** | `src/rsus/repair.py` caches full fp32 entry distributions and computes teacher-forced answer-token KL; `src/rsus/stage2.py` remains explicitly legacy |
| 4.4 Eq. (8) filter | columns of `G` are gradients of active oriented constraints `c_j`; refresh every accepted `m_ref` steps; fixed ridge `lambda` | **implemented under an explicit conservative reduction; not campaign-integrated** | `src/rsus/repair.py` uses one `c_j` per frozen token and example-average margin; this convention must be confirmed in the paper |
| 4.4 acceptance | every forget/neutral/utility token and every example-average margin; rollback, halve, retry up to `J_retry`; stop on exhaustion | **implemented and unit-tested in new core** | `src/rsus/repair.py`, `tests/test_repair_v4.py` |
| 4.4 budget/final output | charge all repair-time model tokens; last saved checkpoint satisfying direct, paraphrase, extraction/generation, and utility contract | **repair token meter/wrapper implemented; paper executor and frozen values absent** | `src/rsus/repair.py` meters root-model forwards plus backward token equivalents; `src/rsus/generators/repaired.py` exposes a separate PDF-v4 wrapper |
| 4.5 sealed protocol | three mutually disjoint target-free folds `D_cal`, `D_pred`, `D_prot`; immutable target freeze; score-independent random/neutral stream; eligibility filtering | **not executable as specified** | paper config declares folds but executors do not consume it; `configs/paper/campaign.yaml:63-81`; protection runner requires its development roster to equal old calibration roster at `alpha_protection.py:474-481` |
| 4.6 RQ1 | LBs for absolute joint rho, gains over S0/S1, positive-damage tail lift, at least 80% tail eligibility coverage | **v4 decision implemented; raw exporter/registry not migrated** | `src/rsus/evidence/pdf_v4.py` fails closed on all four LBs and coverage; legacy `schemas.py`/`raw.py` still omit inputs |
| 4.7 RQ2 | fidelity-floor LBs plus `g_H` and strongest frozen simple-control gain, with validity/common support | **v4 decision implemented; evidence producer absent** | `src/rsus/evidence/pdf_v4.py` defines the separate four-way IUT; fidelity runner remains diagnostic and does not populate it |
| 4.8 RQ3 | four comparators × mean/CVaR superiority (8 UCBs), four native-metric NI LBs, all-arm feasibility/common support | **12-way v4 decision implemented; exporter/registry still 8-way** | `src/rsus/evidence/pdf_v4.py` requires all twelve bounds; legacy raw protection schema has no native metric |
| 4.9 breadth/failure | RQ1, RQ2, and RQ3 remain separate; complete denominators; within-readout breadth correction | **denominator shell partly aligned; claim structure stale** | `configs/paper/evidence.yaml:11-22`, `src/rsus/evidence/decisions.py:191-284` only handle two claims |

## Critical implementation mismatches

### P0 — results would answer a different question

1. **Equation (7) is absent from the active campaign path.** Legacy
   `run_stage2` never computes the entry-anchored neutral KL. The new
   `repair.py` core does, but no paper-stage executor selects that path yet.

2. **The active campaign's Equation (8) basis is wrong.** Its legacy basis is
   `{grad(-L_forget), grad(-L_remote)}`. The new core uses every active token
   and example constraint, but the PDF must confirm that scalar reduction.

3. **The active campaign's hard guard has different semantics.** Legacy code
   accepts a mean squared one-sided forget penalty at periodic refreshes. The
   new core implements the PDF's maximum-style token/example checks and
   same-step rollback/retry, but is not yet wired into the executor.

4. **No processed-token budget is enforced by the active campaign.** The new
   core meters actual root-model forwards (including checkpoint callbacks),
   backward token equivalents, and stops before exceeding `B_tok`. The legacy
   runner and parent objective telemetry remain incomplete.

5. **The neutral stream was score-dependent in construction.** This has been
   corrected in the working tree: `R0` is now sampled before profile scoring,
   removed from repair eligibility, and passed unchanged to an exact-Top-`Kp`
   constructor. The broader duplicate/paraphrase/template exclusion manifest
   is still missing, so the worker records its eligibility status as
   provisional and remains non-claim-bearing.

6. **The primary gate driver uses terminal damage.** `gate.py` calls
   `rec.damage_at()` without selecting the first reaching checkpoint. This is
   not `d_{t dagger}`. The alpha-protection parent branch is the correct model
   for this part.

7. **The active RQ1 ledger is under-specified.** It stores point estimates for joint rho
   and top-q recall, but the decision tests only the two endpoint gains. It
   omits the absolute joint-rho lower bound, positive-damage tail lift and its
   lower bound, eligibility coverage, and strongest-control support.

8. **RQ2 has no active claim path.** Exact-energy fidelity is emitted as an
   appendix measurement. A new isolated v4 decision combines
   `f_rho - 0.80`, `f_K - 0.70`, `g_H`, and `g_ctl`, but the raw producer and
   registry do not populate or expose it yet.

9. **The active RQ3 path omits all four native-metric non-inferiority tests.** The exporter
   writes candidate damage and feasibility slacks only. The decision can pass
   on eight damage contrasts even though the PDF requires twelve favorable
   one-sided bounds.

10. **CVaR.95 was calculated incorrectly at non-integral boundaries.** The
    working tree now uses exactly 5% empirical mass with fractional boundary
    weight in the shared analysis/evidence path and the two legacy campaign
    aggregators. A 30-value regression case fixes the 1.5-observation formula.

11. **The paper campaign has no compliant executable path.** The config
    intentionally points calibration/prediction to a legacy launcher that
    lacks the paper stage contract, several datasets/rosters are `TBD`,
    Llama is unprovisioned, and the selection freeze is absent/draft.

12. **The protection runner and paper roster disagree.** The legacy TOFU
    alpha config runs only `graddiff` and `rmu`, while the PDF and paper
    campaign require seven primary parents. It also scores discovery only, so
    it cannot emit the sealed audit profile needed by RQ1.

13. **Semantic repair eligibility is not implemented.** The new exact-Top-`Kp`
    constructor accepts and hashes an explicit eligibility set, but no frozen
    generator yet excludes direct duplicates, paraphrases, and template-derived
    restatements. The alpha worker therefore labels its provisional mask as
    non-claim-bearing.

14. **The checked-in paper source describes the superseded method.** In
    `paper/sections/04_method.tex:87-144`, endpoint channel routing remains the
    primary method and interior alpha is called diagnostic-only. Lines
    `146-174` describe the old paired hinge guard and score-specific remote
    pool. `prereg/constants.yaml` freezes the same older endpoint/crossed
    protocol. Rebuilding that LaTeX would not recreate the supplied PDF.

### P1 — important validity or reproducibility gaps

- `knn_feature` silently clips `k` to the forget-set size. A frozen contract
  should reject an invalid `k` instead of changing the estimand.
- The production random direction is sampled in parameter dtype. This is safe
  only because confirmatory configs currently require fp32; a new low-precision
  model must not inherit it silently.
- Parent trajectory cost telemetry records wall time but not optimizer forward,
  backward, or token counts. Repair budget comparisons are therefore not
  auditable.
- TOFU's native retain audit is explicitly empty in the adapter, so current
  TOFU output cannot satisfy RQ3's native-metric requirement.
- MUSE code exists but is not registered in the paper adapter registry; WMDP
  and PISTOL are absent; several paper rosters remain unresolved.
- The per-request SFT construction of `theta0` is operationally important but
  absent from the PDF contract. The current runner fine-tunes forget plus the
  request-specific candidate universe, producing a different starting model
  per request/seed. That must either be declared and hashed as the intended
  estimand or replaced by a common benchmark checkpoint.
- New CPU tests cover Eq. (7) reference distributions, the damped filter,
  fail-closed `B_tok`, exact Top-`Kp`, and hard-feasible accepted updates.
  Retry-exhaustion edge cases, GPU numerics/memory, RQ1 tail eligibility, RQ2
  IUT, and native NI still lack end-to-end tests.

## Problems in the PDF itself

The PDF is a coherent new protocol at the conceptual level, but it is still a
draft rather than a fully executable specification.

### Blocking omissions

1. **Result and contract placeholders remain.** The abstract/body contain
   `[Report ...]` and `[Insert ...]`; Figures are referenced as `Figure ??`;
   Tables 1--6 contain dashes rather than results. This is editorial, but it
   means the PDF is not a completed paper.

2. **Table 3 omits the exact gate and utility definitions.** Direct forgetting,
   paraphrase forgetting, general utility, and their boundaries are `--`.
   The text delegates them to a frozen manifest, but no complete manifest is
   included in the PDF or current repository. Consequently Eq. (1), final
   feasibility, and the reported slacks cannot be reproduced uniquely.

3. **Equation (7)'s KL reduction is ambiguous.** For an autoregressive
   prompt-answer behavior, `KL(p_entry(.|x) || p_theta(.|x))` must say whether
   it is the mean teacher-forced answer-token conditional KL, a sequence-level
   KL, or another reduction; it must also specify masking and length weighting.
   These choices yield different gradients.

4. **The differentiable constraints `c_j` are not enumerated.** Appendix D.1
   defines hard per-token/per-example adverse margins, but it never maps them
   to the scalar differentiable `c_j` columns used by Equation (8). An
   implementation cannot know whether `G` contains per-token constraints,
   per-example means, smooth maxima, or aggregate surrogates.

5. **`B_tok` accounting is not mathematically operational.** The PDF lists the
   categories charged, but not whether backward tokens have unit/equivalent
   weight, how autoregressive generation tokens are counted, or whether a step
   that crosses the budget is evaluated, rolled back, or accepted.

6. **Native metrics and NI margins are unspecified.** RQ3 requires four native
   non-inferiority bounds, but Table 3 does not name each dataset metric,
   orientation, or `delta_nat`. That makes the twelve-way IUT incomplete.

7. **`theta0` preparation is missing.** The PDF names model families but does
   not state whether each starting model is a released unlearning checkpoint,
   a full-dataset fine-tune, or the current request-specific SFT used by code.

8. **Outcome tie handling is incomplete.** Candidate IDs are allowed only for
   the last Top-`Kp` allocation boundary, while RQ1 also needs a top-`m`
   positive-damage set. The PDF does not define fractional/randomized handling
   for a damage tie at that boundary.

### Mathematical clarifications worth making

- Equation (4) is correct as a fixed-dimensional asymptotic statement, but its
  `O(eta^2)` remainder suppresses dependence on block dimension, gradient size,
  and the third-derivative bound. State those regularity/uniformity conditions
  if the identity is used as more than a ranking motivation.
- With `lambda > 0`, the matrix in Equation (8) is a damped filter, not an exact
  orthogonal projector: it is not idempotent and does not make `G^T g_bar`
  exactly zero. The prose mostly acknowledges damping, but “projected” should
  not imply exact tangency. The hard acceptance test is what preserves the
  finite-step contract.
- Give the empirical CVaR formula explicitly. For sorted descending damages
  `z_(1) >= ... >= z_(n)` and tail mass `a=0.05n`, the intended statistic is
  `(sum_{i=1}^{floor(a)} z_(i) + (a-floor(a)) z_(floor(a)+1)) / a`, with the
  maximum convention when `a < 1`.
- Freeze `k <= |Df|` rather than allowing an implementation-dependent nearest-
  neighbor fallback.

## Verification performed

- All 13 PDF pages were extracted and inspected, including equations (1)--(9),
  Sections 4.1--4.9, and Appendices A, C, and D.
- Relevant runner, probe, repair, partition, evidence, dataset, configuration,
  and test files were statically traced.
- `python3 -m compileall -q src experiments tests` passes.
- An isolated CPU runtime under `/tmp` collected 209 tests: 207 passed and two
  pre-existing tests were skipped. This includes the new v4 repair, exact
  Top-`Kp`, neutral independence, and fractional-CVaR regressions. No GPU/model
  campaign was run, so large-model numerics and memory remain uncertified.
- Paper preflight exits 2 with `0/8` settings and `0/32` stages ready, as it
  should: model paths, v4 executors, selection freeze, and several adapters/
  rosters are unresolved.

## Required rebuild order

1. Freeze the missing PDF definitions: KL reduction, exact `c_j`, token-cost
   units, dataset native metrics/NI margins, `theta0`, and tie policy.
2. Confirm the new core's scalar-constraint convention, freeze its parameters,
   and integrate it (not legacy `stage2.py`) into the paper executor.
3. Freeze neutral/guard/utility streams and repair eligibility before scoring.
4. Build one paper-stage executor that consumes `D_cal`, `D_pred`, `D_prot`,
   and target rosters without translating through legacy configs.
5. Migrate the raw exporter/registry atomically to the tested v4 RQ1, RQ2, and
   RQ3 decisions and populate dataset-native NI effects.
6. Update LaTeX, preregistration, README, tests, and configuration together;
   then run CPU tests, tiny-model integration, numerical fidelity, and a
   target-free dry campaign before any target run.
