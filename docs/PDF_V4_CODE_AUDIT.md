# PDF Chapter 4 to code audit

Audit date: 2026-07-24
Remediation update: 2026-07-25
Normative specification: `KDD_UnlearningFail.pdf` (13 pages)

## Bottom line

The active code contract is aligned with the July 24 PDF at the metric,
repair, evidence-schema, and Table 1 column level. It is not aligned at the
setting-roster level. The PDF declares TOFU primary/scale/family as
Qwen2.5-1.5B/Qwen2.5-7B/Llama-3.1-8B and Table 2 has eight rows; the active
config instead promotes 7B to primary and adds 1.5B-boundary plus 14B to a
nine-row denominator. This remediation does not create experimental results
and does not make the repository claim-ready.

Legacy channel-matrix and Stage1/Stage2 programs remain diagnostic tools. They
are not accepted as paper evidence. The paper path is:

```text
campaign.yaml
  -> preflight.py
  -> dataset PAPER_UNIT_CONTRACT producer
  -> run_v4_stage.py
  -> aggregate_raw.py
  -> build_evidence.py
```

## Compatibility matrix

| PDF requirement | Remediated code status | Evidence |
|---|---|---|
| Normalized symmetric loss-shake and fp32 confirmatory execution | Implemented | `src/rsus/probe/finite_diff.py`, `src/rsus/probe/fidelity.py` |
| Hidden proximity and separately frozen `alpha_pred`/`alpha_prot` | Implemented | `src/rsus/probe/baselines.py`, `src/rsus/analysis/mixture.py` |
| Damage at first saved direct-criterion-reaching checkpoint | Implemented; non-reaching runs fail closed | `src/rsus/generators/base.py`, `experiments/gate_1p5b/gate.py`, `src/rsus/local_pdf_v4.py` |
| Equation (7): protected loss plus entry-distribution KL on neutral data | Implemented | `src/rsus/repair.py` |
| Equation (8): damped active-constraint filter, refresh, rollback and retry | Implemented and unit-tested | `src/rsus/repair.py`, `tests/test_repair_v4.py` |
| `B_tok` repair budget and final feasibility gates | Implemented in v4 repair wrapper | `src/rsus/repair.py`, `src/rsus/generators/repaired.py` |
| Exact `Kp`, selector-independent neutral data and score-independent folds | Implemented and frozen in manifests/plans | `src/rsus/partition.py`, `src/rsus/local_pdf_v4.py`, `experiments/paper/init_raw_plan.py` |
| Exact `D_cal`, `D_pred`, `D_prot`, target roster consumption | Orchestrator implemented for its config; active setting roster does not match the latest PDF | `experiments/paper/run_v4_stage.py`, `configs/paper/campaign.yaml` |
| Dataset model execution into paper raw schemas | Implemented for TOFU; other datasets remain blocked | `experiments/paper/tofu_v4_unit.py`, `configs/paper/campaign.yaml` |
| RQ1 four lower bounds and at least 80% tail eligibility | Implemented in schema-v2 raw aggregation and decision gate | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| RQ2 fidelity/add-value four-way IUT | Implemented | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| RQ3 eight damage UCBs and four native-NI LBs | Implemented; native raw data is mandatory | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| Fractional empirical CVaR.95 | Implemented and regression-tested | `src/rsus/analysis/prediction.py`, evidence raw tests |

## Resolved mismatches

1. The active paper stages no longer point at `channel_matrix` launchers.
   `run_v4_stage.py` validates the exact parent/request/seed Cartesian roster,
   resolves the configured adapter, executes shell-free argv commands, checks
   each shard, and writes consolidated JSONL plus SHA-256 provenance.

2. The evidence ledger is schema version 2 with separate `rq1`, `rq2`, and
   `rq3` blocks. The old two-claim registry and decision path have been
   replaced atomically.

3. RQ1 now requires favorable lower bounds for absolute joint rho,
   joint-minus-S0, joint-minus-S1, and positive-damage tail lift. `tail_m` is
   frozen per plan unit and fewer than 80% eligible reached units cannot pass.

4. RQ2 now consumes one fidelity record per unit and requires the four PDF
   effects plus perturbation, exact-reference, and common-control validity.

5. RQ3 now requires candidate-level native retention for every arm and draw.
   A row cannot pass without all five arms, every repeated-random draw, common
   candidate support, nonnegative feasibility slacks, eight favorable damage
   UCBs, and four favorable native-NI lower bounds.

6. Paper protection rows must identify one first-reaching parent checkpoint.
   The stage executor and raw parser both reject terminal-budget substitution.

7. Every dataset config now declares a native metric name, orientation, and
   frozen non-inferiority margin. Missing values block plan creation and
   preflight.

8. MUSE-News and MUSE-Books loaders are registered, but their capability
   correctly says that the current corpus-level request does not provide
   independent target-request rosters. Preflight does not treat registration
   alone as paper readiness.

9. Stage orchestration and dataset model execution are now separate contracts.
   `run_v4_stage.py` cannot satisfy `PAPER_UNIT_CONTRACT`; TOFU uses the real
   model producer in `tofu_v4_unit.py`.

10. TOFU now has a complete freeze-ordered workflow:
    `run_tofu_table1.py` runs parent calibration, prediction/control selection,
    protection selection, target execution, raw aggregation, and Table 1
    rendering. Non-differentiable forgetting/native constraints participate in
    tentative-update rollback through `external_feasibility`.

## Remaining blockers

These are real missing inputs or experiments, not code paths that may be
silently substituted:

- The configured `/group-volume/models/...` paths are absent on this host.
- Non-TOFU datasets still need their own `PAPER_UNIT_CONTRACT` producers.
- Llama-3.1-8B is not provisioned.
- `configs/paper/selection_freeze.yaml` does not exist until the implemented
  TOFU `D_pred`/`D_prot` selector has consumed real development artifacts.
- `configs/paper/tofu_parent_freeze_1p5b.yaml` remains draft until real
  `D_cal` results resolve all seven parents.
- WMDP-bio/MMLU and PISTOL lack real adapters and exact request rosters.
- MUSE needs defensible independent deletion-request semantics before its four
  rosters can be frozen.
- WMDP, MUSE, and PISTOL roster placeholders remain unresolved.
- No full GPU campaign has produced the candidate-level RQ1/RQ2/RQ3 shards.
- The active evidence config has nine setting rows, while the PDF Table 2 has
  eight; primary/scale roles and the extra 14B row must be reconciled to the
  PDF before a claim-bearing run.
- The checked-in LaTeX predates the supplied PDF; the generated current-format
  Table 1 is written separately because the PDF's matching source tree is not
  present in this repository.

## Specification gaps

The PDF itself still leaves values that must be frozen before a confirmatory
run:

- dataset-specific native metrics and scientifically justified NI margins;
- exact direct, paraphrase, extraction/generation, and utility boundaries;
- the intended `theta0` preparation and checkpoint provenance;
- the final semantic duplicate/paraphrase eligibility review;
- operational token-equivalent accounting if backward work is weighted other
  than the implemented convention;
- tie policy for the positive-damage top-`m` outcome boundary.

The implementation resolves Equation (7) as mean teacher-forced answer-token
KL against cached full fp32 entry distributions. Equation (8) uses one
oriented constraint per frozen token plus example-average margins and a fixed
ridge. These conventions are explicit in code and must be stated in the paper
or changed before a confirmatory run.

## Verification boundary

CPU tests cover v4 repair, exact `Kp`, score-independent manifests, first-reach
provenance, stage-roster sealing, RQ1/RQ2/RQ3 aggregation, tail eligibility,
native non-inferiority, fractional CVaR, registry validation, and preflight.

`experiments/paper/preflight.py` is expected to exit 2 in the current checkout.
`summary.unready_executors` is empty because the stage orchestrator is real;
`summary.unready_unit_producers` remains nonempty until dataset model runners
exist. No GPU/model campaign was run as part of this remediation.
