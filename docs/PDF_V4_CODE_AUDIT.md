# Current paper/code audit

Audit basis: current `main` branch.

## Source of truth

The active contract is defined jointly by:

```text
paper/
configs/paper/evidence.yaml
configs/paper/campaign.yaml
src/rsus/evidence/tables.py
```

The checked-in `KDD_UnlearningFail.pdf` is an older snapshot. It must not be
used to replace the current 7B-primary, nine-setting contract with its older
1.5B-primary, eight-row layout.

## Active roster

- Primary: TOFU / Qwen2.5-7B
- Scale boundary: TOFU / Qwen2.5-1.5B
- Model scale: TOFU / Qwen2.5-14B
- Model family: TOFU / Llama-3.1-8B
- Dataset replication: WMDP-bio/MMLU, MUSE-News, RWKU
- Stress: MUSE-Books, PISTOL
- Parents: GradDiff, NPO, SimNPO, GRU, RMU, RepNoise, Circuit Breakers

All nine settings and all seven parents remain in the denominator even when
their evidence is missing, invalid, infeasible, or non-reaching.

## Implemented contract

| Requirement | Current implementation |
|---|---|
| Forward-only Loss Susceptibility and exact-gradient reference | `src/rsus/probe/finite_diff.py`, `src/rsus/probe/fidelity.py` |
| Representation Proximity and frozen joint mixture | `src/rsus/probe/baselines.py`, `src/rsus/analysis/mixture.py` |
| First-reaching parent checkpoint | `src/rsus/generators/base.py`, campaign runners |
| Fixed-budget repair, KL, active constraints, rollback | `src/rsus/repair.py` |
| RQ1 rank and harmful-tail evidence | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| RQ2 fidelity and added-value evidence | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| RQ3 damage, native NI, feasibility evidence | `src/rsus/evidence/raw.py`, `src/rsus/evidence/pdf_v4.py` |
| Shared ledger merge and atomic LaTeX publish | `experiments/paper/publish_evidence.py` |
| Current core and robustness tables | `src/rsus/evidence/tables.py` |

The current core output contains five tables: prospective rank prediction,
Loss Susceptibility fidelity, harmful-tail recovery, repair effects, and the
repair contract. Robustness output contains claim breadth and the evidence
funnel.

## Execution paths

Existing 7B results can be converted to the current paper schema without GPU
work:

```bash
bash experiments/cluster/run_tofu_7b_h100.sh render-only
```

The full 7B H100 campaign uses explicit experiment mode:

```bash
bash experiments/cluster/run_tofu_7b_h100.sh experiment
```

The 14B launcher currently produces diagnostic CSV/JSON aggregates. The 1.5B
4090 pipeline is a scale-boundary experiment and does not define the current
primary setting.

## Remaining evidence gaps

These are missing executions or dataset producers, not missing table rows:

- 14B has no committed paper prediction-alpha freeze and remains
  diagnostic-only.
- Non-TOFU settings do not yet have complete dataset-specific
  `PAPER_UNIT_CONTRACT` producers.
- A setting without fidelity evidence keeps RQ2 cells explicitly ineligible.
- A parent without complete five-arm repair evidence keeps RQ3 cells
  explicitly ineligible.
- Missing settings remain visible as placeholders in the nine-setting
  robustness denominator.

The 7B backfill exports all seven parent rows. GradDiff and RMU use their
frozen prediction selections. Other parents use explicitly declared
descriptive selections and remain claim-ineligible unless a valid frozen
selection is supplied. A non-reaching parent may display its last completed
checkpoint descriptively, but it cannot license a claim.

## Verification boundary

A generated `.tex` file proves that rendering completed; it does not prove
that the claims passed. Use:

```text
/group-volume/fdmu/runs/paper_v4/evidence_readiness.json
/group-volume/fdmu/runs/paper_v4/PUBLISH_STATUS.json
```

`Rank E/P=y/y` establishes only the rank condition. Full RQ1 additionally
requires harmful-tail recovery, so claim success requires both `Rank E/P`
and final `RQ1 E/P` to be `y/y`.
