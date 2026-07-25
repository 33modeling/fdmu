# Paper evidence pipeline

`experiments/paper/build_evidence.py` is the only paper-facing result gate.
Experiment runners may write many diagnostic artifacts, but the paper consumes
only a normalized ledger checked against `configs/paper/evidence.yaml`.

The config is the denominator. Every configured setting--parent row appears in
the readiness report even if it was never attempted. A row cannot pass while a
planned trajectory is incomplete, a profile is invalid, a development weight
is unresolved/fallback, common support is missing, or a protection arm is
infeasible. Conditional estimates may still be inspected, but they never
license prose.

## Run

From the repository root:

```powershell
python experiments/paper/build_evidence.py
python experiments/paper/build_evidence.py --paper-root ../paper --require-ready
```

The first command always writes `results/paper/evidence_readiness.json` and is
useful while jobs are running. The second additionally verifies every completed
artifact's SHA-256 and atomically replaces
`<paper-root>/sections/generated/results_macros.tex`. It returns exit status 2
until all registered tables have complete data. Invalid schemas, hashes, or
unregistered rows return exit status 1 and do not touch the paper macro file.

The generated file owns exactly these commands:

- `\TailHeadline`
- `\PredictionHeadline`
- `\FidelityHeadline`
- `\ProtectionHeadline`
- `\TransferHeadline`

An incomplete evidence block remains a visible `\resph{...}` placeholder.

## Normalized row contract

The ledger has `schema_version: 2`, a `rows` list, and an `artifacts` mapping.
Each row is keyed by the predeclared `setting` and `parent`. Its funnel records:

```text
profiles_valid <= profiles_planned
fidelity_valid <= fidelity_planned
trajectories_reached <= trajectories_completed
  <= trajectories_attempted <= trajectories_planned
prediction_common <= reached_with_valid_profile <= trajectories_reached
tail_eligible <= prediction_common
fidelity_common <= min(fidelity_valid, prediction_common)
protection_common <= protection_feasible_all_arms
  <= reached_with_valid_profile
```

`completed: true` additionally requires all planned trajectories to have been
attempted and completed. RQ1 supplies lower bounds for absolute joint rho,
joint-minus-`S0`, joint-minus-`S1`, and positive-damage tail lift, plus at
least 80% tail eligibility. RQ2 supplies lower bounds for `f_rho - 0.80`,
`f_K - 0.70`, `g_H`, and `g_ctl`, with perturbation, exact-reference, and
common-control validity. RQ3 supplies one-sided upper bounds for all four
comparators (`no_repair`, `repeated_random`, `s0`, `s1`) crossed with `mean`
and `cvar95`, and one native-metric non-inferiority lower bound per comparator.
All three blocks must explicitly set `paired: true`; omitting it fails closed.

Completed non-row artifacts require `path`, `sha256`, and may provide a
validated `headline_tex`. Relative paths are resolved against this repository.

## Table registry

The registry covers two main-paper tables and five appendix evidence blocks
across four physical appendix tables (budget and specificity share one float):

| Registry ID | Paper label | Required evidence |
|---|---|---|
| `main_core_evidence` | `tab:core-evidence` | primary RQ1 + RQ2 + RQ3 rows |
| `main_robustness` | `tab:robustness` | all predeclared RQ1/RQ2/RQ3 rows |
| `appendix_scope_contract` | `tab:datasets` | frozen campaign manifest |
| `appendix_tail_prediction` | `tab:tail-structure` | primary RQ1 + tail artifact |
| `appendix_lse_fidelity_cost` | `tab:bwfree` | LSE fidelity/time/memory artifact |
| `appendix_protection_budget` | `tab:budget-sweep` | primary RQ3 + budget sweep |
| `appendix_boundaries` | `tab:specificity` | all rows + negative controls |

Readiness and claim success are separate. A fully observed null or failed IUT
makes a table data-ready while correctly leaving the scientific claim failed.
The setting-level rule is also explicit: at least one output-readout and one
representation-readout parent must each pass all three claims after a Bonferroni
correction within its predeclared parent group; the other parent rows remain
reported denominators. The multi-setting statement then requires
the primary setting, at least one of two model-transfer settings, and at least
two of three dataset replications. Stress settings can never rescue the rule.

## Candidate-level raw aggregation

`experiments/paper/aggregate_raw.py` is the shared CPU path from all dataset
adapters to the normalized ledger. Dataset runners write JSONL records; one
immutable JSON plan fixes every `(setting, parent, request, seed)` denominator
and repeated-random draw roster before results are inspected.

Create that plan only after development selections are frozen:

```powershell
# First run writes a deliberately invalid draft; fill every selected parent.
python experiments/paper/init_raw_plan.py `
  --write-selection-template configs/paper/selection_freeze.yaml

# Set status: frozen, a final non-PENDING freeze_id, and
# frozen_before_target: true, then freeze the executable denominator.
python experiments/paper/init_raw_plan.py `
  --selection-freeze configs/paper/selection_freeze.yaml `
  --out results/paper/raw_plan.json
```

`init_raw_plan.py` rejects unresolved target rosters, unprovisioned models,
missing parent implementations, duplicate seeds/draws, a draft freeze ID, and
fidelity settings that the cost runner cannot execute. A fallback selection
can be frozen explicitly but remains claim-ineligible. For development of one
ready setting, repeat `--setting SETTING_ID`; this is a partial execution plan,
not evidence for omitted settings.

The fidelity artifact keys include model ID and frozen source path, precision, the exact block regex,
the SHA-256 of every frozen runner argument, profiler, direction count, and
repeat. Therefore the cost runner must pass the
campaign model key explicitly (its path-basename default is intentionally not
accepted). For the current Qwen2.5-7B cell:

```powershell
python experiments/cost/bench.py `
  --model /group-volume/models/Qwen2.5-7B-Instruct `
  --model-id Qwen2.5-7B --device cuda --dtype float32 `
  --author 198 --candidate-authors 0-29 --n-candidates 128 `
  --candidate-seed 314159 --block-last-n 8 --norm-eta 0.003 `
  --dirs 16,32,64 --batch-size 4 --repeats 3 --seed 0 --k 0 `
  --min-rho 0.8 --min-overlap 0.7 --min-split-half 0.7 `
  --min-perturbation-survival 0.9 `
  --out runs/paper/lse_qwen25_7b.jsonl
```

Use the identical execution block for Qwen2.5-1.5B, changing only `--model`,
`--model-id Qwen2.5-1.5B`, and the output filename. The raw key validator also
checks precision, the runtime-emitted block regex, and the protocol hash, so a
wrong dtype, radius, threshold, roster seed, or `--block-last-n` cannot fill
the planned cell.

```powershell
python experiments/paper/aggregate_raw.py `
  --plan results/paper/raw_plan.json `
  --prediction-raw runs/tofu/prediction.jsonl `
  --fidelity-raw runs/tofu/fidelity.jsonl `
  --protection-raw runs/tofu/protection.jsonl `
  --artifact-raw lse_fidelity_cost=runs/paper/lse.jsonl `
  --artifact-raw protection_budget_sweep=runs/paper/budget.jsonl `
  --artifact-raw specificity_negative_controls=runs/paper/specificity.jsonl `
  --artifact-raw tail_structure=runs/paper/prediction_supplement.jsonl `
  --out results/paper/evidence_ledger.json
```

An absent shard is allowed, but its unit remains planned and makes its row or
artifact incomplete. An extra unit, duplicate candidate, changed frozen
selection, undeclared random draw, duplicate measurement key, or raw artifact
without a plan contract is an error. The command prints the consumed plan's
SHA-256.

### Immutable plan and candidate records

The core plan has `schema_version: 2`, bootstrap fields (`replicates`, `seed`,
`alpha`, `top_q`, `cvar_q`), and a `units` list. Each unit contains its four
keys, frozen prediction/protection selections, and exact
`simple_control_name`, `repeated_random_draws`, positive-damage tail size
`tail_m` (the frozen `Kp`), and the dataset's frozen native metric name,
orientation, and non-inferiority margin. Selections cannot vary within a
setting--parent row.

Prediction JSONL has one row per candidate and unit: the four unit keys,
`candidate_id`, semantic `group`, `s0`, `s1`, `joint`, `simple_control`,
`simple_control_name`, `damage`, `profile_valid`, `reached`,
`trajectory_completed`, and the frozen selection.
Every row identifies its parent checkpoint, and reached rows certify that it is
the first checkpoint satisfying the direct criterion.
The aggregator forms all correlations on identical candidates, averages seeds
within requests and requests equally, then bootstraps requests, seeds, and
semantic groups. Missing support is not repaired by intersection.

Fidelity JSONL has exactly one row per unit with `f_rho`, `f_k`,
`perturbations_valid`, `exact_reference_valid`, and
`common_control_support`. These records feed only RQ2.

Protection JSONL has one row per candidate, arm, and unit, with `damage`, the
dataset-native `native_retention`, `native_metric_name`, the frozen selection,
exact `Kp`, and four explicit slacks: `direct_forget_margin`,
`paraphrase_forget_margin`, `extraction_generation_margin`, and
`utility_margin`. `feasible` must equal their conjunction; a missing metric or
an inconsistent flag is rejected. Every row also carries the same
`parent_checkpoint_id` and `parent_checkpoint_first_reaching: true`, proving
that all arms branch from one first criterion-reaching parent state. Ordinary
arms omit `draw_id`; `repeated_random` supplies every planned `draw_id` and
`draw_complete`. The five arms must have exact candidate/group support.
Feasibility and common support remain separate funnels: fully feasible arms
with mismatched candidates cannot enter a paired effect. The eight
mean/CVaR95 effects and four native-NI effects are paired. Inside each bootstrap replicate, random draw
IDs are themselves resampled with replacement before averaging; a missing or
incomplete draw makes the unit infeasible and non-common.

### Runner-to-ledger boundary

The historical channel-matrix JSON (`results.json`) is not the candidate JSONL
schema above and is never consumed directly by the paper gate. The stage
orchestrator is `experiments/paper/run_v4_stage.py`. A separate dataset unit
producer must create the shards; the orchestrator only runs that frozen
command, validates its outputs, and seals them. A frozen stage manifest contains
the exact `(parent, request, seed)` Cartesian roster, a shell-free argv command
per unit, and its output shard paths:

```powershell
python experiments/paper/run_v4_stage.py `
  --manifest configs/paper/stages/tofu_target.yaml `
  --output-dir runs/paper/tofu_target `
  --action run
```

The executor resolves the configured adapter, rejects missing/extra units,
runs every command without a shell, checks first-reaching provenance, exact
five-arm and repeated-random support, fidelity validity, and native retention,
then writes consolidated JSONL and SHA-256 manifests. `--action verify` seals
already-produced shards; `--action validate` checks only the frozen denominator.
The calibration stage emits fidelity rows plus sealed parent-selection inputs,
`D_pred` emits prediction rows plus target-free `alpha_pred` selection inputs,
and `D_prot` emits protection rows plus target-free `alpha_prot`/`Kp`
selection inputs. The target stage emits the three claim-facing raw streams
with their frozen selection mappings.
Every stage also validates each unit's `run_manifest.json`, including raw,
profile, score-independent split, diagnostic, config, parent-freeze, and
target selection-freeze hashes before writing the sealed stage manifest.

TOFU implements that producer in `experiments/paper/tofu_v4_unit.py`.
`experiments/paper/run_tofu_table1.py --action run` executes the four stages in
freeze order, creates the partial TOFU raw plan, aggregates the three target
streams with `--core-only`, and writes the two-panel claim-bearing table to
`paper/sections/generated/table1.tex`. `--action plan` creates exact manifests
without opening target outcomes. The 1.5B parent freeze remains deliberately
draft until complete `D_cal` artifacts exist.

### Schema-backed appendix artifacts

Artifacts are generated beside the ledger and marked complete only after
their exact planned keys validate. Incomplete diagnostic JSON is written with
`completed: false`, so it cannot license a table or headline.

- `campaign_manifest` freezes every scope and feasibility table column.
- `tail_structure` derives damage concentration, semantic-group lift,
  hierarchical intervals, and permutation p-values from candidate rows. Its
  contracted `reference_roster` and `construction_checks` blocks are also
  mandatory, so Panels B/C cannot remain unbacked while Panel A is complete.
- `lse_fidelity_cost` requires fidelity/overlap, split-half and perturbation
  checks, synchronized time, peak memory, integrity, and backward-access cells.
- `protection_budget_sweep` requires worst effect/UCB, bottleneck,
  eligibility/pass, margins, accepted updates, common support, and random-draw
  completeness for every budget--parent cell.
- `specificity_negative_controls` requires the three correlations, top-q
  lift, displacement matching, and common support for every motion cell.

Measurement contracts declare `key_fields`, `group_by`, exact `planned` keys,
and typed metrics with deterministic aggregation. Output JSON retains every
cell's `source_keys`, providing cell-level provenance rather than only an
arbitrary file hash.

## Physical table source map

| Physical table/block | Cell or decision | Producer/source |
|---|---|---|
| Main `tab:core-evidence` | prediction effects/LBs | all-common candidate prediction rows |
| Main `tab:core-evidence` | protection effects/UCBs, margins, funnels | exact five-arm candidate protection rows |
| Main `tab:robustness` | coverage and least-favorable bounds | normalized ledger plus claim decisions; missing rows stay denominators |
| Appendix contract | scope and feasibility | immutable campaign manifest |
| Appendix tail/reference/construction | tail Panel A | candidate `damage` and semantic `group` |
| Appendix tail/reference/construction | Panels B/C | contracted supplementary raw keys |
| Appendix fidelity/cost | all cells | contracted LSE measurements and retained source keys |
| Appendix protection/specificity | budget Panel A | contracted budget--parent measurements |
| Appendix protection/specificity | specificity Panel B | contracted motion--conditioning measurements |

The measurement artifacts never manufacture upstream values. GPU runners must
export contracted unit-level measurements, including each budget--parent
bootstrap UCB, before those blocks become ready. Core and tail statistics are
computed directly here; fidelity instrumentation, accepted-update logs,
budget summaries, and motion controls remain outputs of their named runners.
