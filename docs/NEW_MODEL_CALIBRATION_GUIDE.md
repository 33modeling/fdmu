# New environment and new LLM calibration guide

This guide follows the July 24 PDF contract, not the superseded endpoint-router
method in the checked-in LaTeX. Read `PDF_V4_CODE_AUDIT.md` first. The Equation
(7)--(8) core and a fail-closed single-request local executor now exist, but a
final target campaign must remain blocked until preflight also certifies the
resolved cross-request freeze, comparator arms, native metrics, and evidence
schema.
Use `configs/paper/new_model_calibration.template.yaml` as the machine-readable
handoff record; it is deliberately non-executable while any value is null or
`TBD`.

## Non-negotiable rule

Never tune on target damage, sealed audit outcomes, target native retention, or
any quantity opened after unlearning. A value is deployable only if it was
chosen on the corresponding target-free fold and committed in an immutable
freeze before the target job starts.

The required order is:

```text
environment/model contract
  -> D_cal: numerical probe and parent operating points
  -> D_pred: alpha_pred
  -> D_prot: alpha_prot and repair/guard operating point
  -> immutable freeze + clean commit
  -> target profile/seal
  -> parent first-reaching checkpoint
  -> repair arms
  -> open sealed outcomes and aggregate RQ1/RQ2/RQ3
```

If any earlier stage is unresolved, do not substitute a convenient default in
the target stage.

## 1. Create and record the environment

Use a fresh virtual environment and pin the full resolved dependency set. The
project requires Python 3.10 or newer; Python 3.11 is the conservative default
for CUDA/Transformers compatibility.

```bash
cd /absolute/path/to/fdmu
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,campaign]'
python -m pip freeze > runs/environment.freeze.txt
python -m pytest -q
```

On Debian/Ubuntu hosts where `python -m venv` reports that `ensurepip` is
missing, install the matching OS package (for example `python3.11-venv`) first.
Do not fall back to mixing packages into the system Python for a claim-bearing
run. A temporary `pip --target` directory is acceptable only for CPU unit-test
diagnostics and must not become the frozen campaign environment.

Record, in the run manifest:

- git commit and clean-worktree status;
- Python, PyTorch, Transformers, CUDA, driver, and GPU versions;
- exact model source revision or local content hash;
- tokenizer source/revision, vocabulary size, BOS/EOS/PAD IDs, chat template;
- dtype, attention implementation, deterministic settings, and device map;
- dataset revision/cache fingerprint and every candidate/guard manifest hash.

The current PDF contract uses fp32 confirmatory profiling. Do not switch to
bf16 merely to fit a model. A low-precision boundary run is diagnostic until a
fp32-shadow perturbation implementation passes the same displacement and
fidelity gates.

## 2. Port the model architecture before tuning

The current block helper matches Qwen/Llama/Mistral-style parameter names:

```text
*.layers.<index>.mlp.down_proj.weight
```

For another architecture, first enumerate `model.named_parameters()` and add a
model-specific `BlockSpec`. Fail if the block is empty or unintentionally
includes parameters outside the declared support. Freeze and report:

- selected parameter names and count `d_B`;
- layer indices and module role;
- whether every parent trains exactly this support or the common intersection;
- parameter dtype and device placement.

Required port tests:

1. save -> `+eta v` -> `-eta v` -> restore is bit-exact;
2. `||v||_2 = 1` in fp32 and all coordinates outside `B` remain unchanged;
3. central-difference response agrees with an exact directional derivative on
   a tiny candidate subset;
4. exact gradient energy and loss-shake use the same loss, masking, batching,
   dropout state, block, and candidate support;
5. hidden-state extraction identifies the intended final layer and answer-token
   positions for the architecture;
6. padding uses the tokenizer's actual PAD ID and no example has zero answer
   tokens after truncation.

Do not start a parameter sweep until these invariants pass.

## 3. Freeze the data and leakage boundaries

For every request, build one immutable candidate manifest before computing a
score. It must include:

- `Df`, `C`, semantic group, `C_disc`, and `C_audit` IDs and hashes;
- repair-eligible flag and exclusion reason;
- direct-duplicate, paraphrase, and template-restatement checks;
- score-independent neutral stream `R0`;
- differentiable guard IDs and general-utility checkpoint IDs;
- native retain audit IDs and metric definition;
- repeated-random draw IDs and seeds.

The neutral stream and repair eligibility are sampled/frozen **before** profile
scores exist. `R0`, guards, utility, audit, and native-audit sets must obey the
disjointness declared by the PDF. A scorer may evaluate an ineligible candidate
for RQ1 profiling, but that candidate may not enter the repair Top-`Kp` pool.

Use four disjoint request rosters:

| Fold | Permitted selection |
|---|---|
| `D_cal` | parent reach/utility settings; `(R, eta)` and numerical integrity |
| `D_pred` | `alpha_pred` only |
| `D_prot` | `alpha_prot`, `Kp`, repair budget, guard/retry settings |
| target | no tuning; only frozen execution and outcome evaluation |

Semantic candidate discovery/audit splitting occurs inside each request and is
not a substitute for these request-level folds.

## 4. Establish `theta0`

Choose exactly one preparation rule and encode it in the manifest. For example:

- a released benchmark fine-tuned checkpoint shared by requests; or
- a deterministic training recipe whose data roster is fixed independently of
  each scored candidate pool.

Do not silently use the current request-specific SFT cache. If that construction
is scientifically intended, state that `theta0` varies by request/seed, freeze
its training data and stopping rule, and hash the complete resulting state.

Before profiling, validate loss and baseline utility at `theta0`, confirm every
required answer has at least one untruncated token, and save the baseline NLLs
used by Equation (2).

## 5. Calibrate the loss-shake probe on `D_cal`

For each candidate `(R, eta)` and declared block:

1. sample normalized shared directions once per frozen direction bank;
2. measure effective displacement `||theta_perturbed-theta0|| / eta` and the
   changed-coordinate fraction;
3. save signed central responses before squaring;
4. compare `qhat_G` with exact `||grad_B ell||^2` on the same candidates;
5. compute Spearman fidelity, Top-`Kp` overlap, direction-split stability, wall
   time, peak memory, and perturbation survival;
6. repeat enough times to form the predeclared one-sided bounds.

Choose the smallest-cost operating point satisfying all frozen integrity floors.
The PDF's confirmatory floors are:

```text
LB(f_rho - 0.80) > 0
LB(f_K   - 0.70) > 0
```

Direction-split and perturbation-survival thresholds are engineering validity
conditions and must also be frozen. The historical `R=64`, `eta=3e-3`, and
last-eight-MLP block in the old TOFU config are starting grid points only. They
are not transferable guarantees for a different architecture, dtype, or block
dimension.

Also freeze representation parameters here: final hidden layer, answer-token
mean pooling, and a valid `1 <= k <= |Df|`. The implementation should reject,
not silently clip, an invalid `k`.

## 6. Calibrate parent unlearners without profile outcomes

For each parent, tune only against its external first-reaching gate and the
frozen ordinary-utility criteria on `D_cal`. Parent objective loss is never a
substitute for `ForgetOK`.

Freeze:

- objective implementation ID/source commit;
- trainable support;
- learning rate, optimizer, coefficients, seed, processed-token budget;
- checkpoint grid `T_save`;
- exact direct-forgetting metric and threshold.

Record non-reaching settings rather than increasing the budget after inspecting
target outcomes. The selected parent must still receive no susceptibility
profile.

## 7. Select `alpha_pred` on `D_pred`

Fit empirical midrank maps on each request's `C_disc`, then apply those frozen
maps to its audit candidates. For every prespecified alpha:

```text
S_alpha = (1-alpha) * qtilde_G + alpha * qtilde_H
```

Run the frozen parents, locate `t_dagger` on `T_save`, and evaluate damage only
for reaching cells. Average seeds within request, then give requests equal
weight. Choose `alpha_pred` by the predeclared development ranking objective and
tie rule. A fallback alpha is diagnostic and makes the target claim ineligible.

Do not use target rho, target tail recall, or any protection outcome in this
selection.

## 8. Select `alpha_prot` and repair parameters on `D_prot`

This is a separate estimand and must not reuse `D_pred`. Each selector chooses
exactly `Kp` repair-eligible discovery examples. Every arm must share the same:

- first-reaching parent checkpoint;
- pre-frozen neutral/guard/utility streams;
- repair code, example order, optimizer seed, and checkpoint schedule;
- processed-token budget;
- hard margins and final feasibility contract.

The rebuilt repair objective is:

```text
L_rep = mean_{x in P} ell_x(theta)
        + beta * mean_{x in R0} KL(p_entry(.|x) || p_theta(.|x))
```

The manifest must fix the exact teacher-forced token reduction for the KL.
Before choosing the neutral-pool size, estimate the fp32 entry-cache footprint
as `4 * answer_tokens(R0) * vocabulary_size` bytes, plus temporary current
logits. The current v4 core stores the exact full distribution; a top-k or
gold-token-only cache changes Equation (7) and is not an allowed memory-saving
substitution. Reduce batch size or use a separately validated frozen-reference
strategy if the exact cache does not fit host RAM.
Tune/freeze on `D_prot` only:

- `alpha_prot` and `Kp`;
- `beta`;
- initial repair step size and optimizer/momentum;
- active margins `kappa_j`;
- accepted-step refresh interval `m_ref`;
- ridge `lambda`;
- token and example tolerances `epsilon_tok`, `epsilon_ex`;
- retry count `J_retry` and halving factor;
- processed-token budget `B_tok` and save schedule;
- dataset-specific final utility/native NI rules.

Selection minimizes worst-request repair CVaR among fully feasible arms, then
applies the frozen tie rule. No feasible alpha means unresolved; it does not
license the least-bad target run.

The historical old-guard values (`eta2=3e-5`, `refresh_k=4`,
`delta_seq_sq=1e-2`, `delta_tok_sq=1e-1`) implement another algorithm. Do not
carry them into the rebuilt Equation (7)--(8) repair by renaming the fields.

## 9. Freeze before target execution

The freeze must contain all resolved values for every model, dataset, and
parent and must be committed from a clean tree. At minimum hash:

```text
environment + code + model + tokenizer
theta0 construction
all four request rosters
candidate/eligibility/neutral/guard/utility/native manifests
block and loss definition
R, eta, direction bank, k
alpha_pred, alpha_prot, Kp
parent configs and T_save
repair beta/kappa/m_ref/lambda/tolerances/retries/B_tok
final feasibility metrics and thresholds
bootstrap/IUT/tie/missingness rules
```

Run the fail-closed check before allocating target GPUs:

```bash
python experiments/paper/preflight.py \
  --campaign-config configs/paper/campaign.yaml \
  --evidence-config configs/paper/evidence.yaml
```

Exit code 2 means “not ready”, not “run anyway”. At present this is the expected
result while the rosters and model-specific selection freeze remain incomplete.

For one local diagnostic, use the narrower fail-closed runner after calibration:

```bash
cp configs/local/pdf_v4.example.yaml configs/local/pdf_v4.local.yaml
bash local_run.sh inspect-model configs/local/pdf_v4.local.yaml
bash local_run.sh prepare-manifest configs/local/pdf_v4.local.yaml
# Fill calibrated nulls and set status: frozen_for_local_diagnostic.
bash local_run.sh validate configs/local/pdf_v4.local.yaml
bash local_run.sh run configs/local/pdf_v4.local.yaml
```

The order matters: `prepare-manifest` must happen before profiling, and the
runner refuses to call the model if the resolved config or frozen manifest is
invalid. Its output is always diagnostic (`claim_eligible: false`); it is not a
shortcut around D_cal/D_pred/D_prot or the full evidence preflight.

## 10. Target execution and acceptance checks

For each target cell:

1. profile at `theta0` and seal audit scores;
2. run the unchanged parent and choose the first reaching saved checkpoint;
3. record non-reaching cells with no conditional prediction/repair outcome;
4. branch every repair arm from the identical checkpoint hash;
5. before each tentative update, construct/refresh `G` only from active frozen
   differentiable constraints;
6. apply the damped Equation (8) direction;
7. check every hard token and example margin; on failure restore parameters and
   optimizer state, halve, and retry up to `J_retry`;
8. charge every protected, neutral, constraint, guard, and checkpoint-evaluation
   token to `B_tok`;
9. choose the last saved checkpoint satisfying the full direct/paraphrase/
   extraction-generation/utility contract without reading sealed outcomes;
10. only then open audit damage and native retention.

## 11. Evidence checks

Do not label a row as passing unless the rebuilt evidence layer verifies:

- **RQ1:** LB(joint rho) > 0, LB(joint-S0) > 0, LB(joint-S1) > 0,
  LB(tail lift) > 0, and tail eligibility coverage >= 0.80;
- **RQ2:** valid perturbations/exact reference/common control support plus
  LB(`f_rho-0.80`) > 0, LB(`f_K-0.70`) > 0, LB(`g_H`) > 0, LB(`g_ctl`) > 0;
- **RQ3:** all eight damage UCBs < 0, all four native NI LBs > 0, every random
  draw complete, and all five arms feasible on identical support.

CVaR.95 must use fractional boundary mass. Correlations stay within a
request-seed-parent cell; seeds are averaged within request; requests receive
equal weight; the hierarchical bootstrap resamples requests, seeds, and semantic
groups, with repeated-random draw IDs resampled and averaged inside their
trajectory.

## 12. New-model handoff checklist

A new LLM is ready only when all answers are “yes”:

- Does the block selector match exactly the intended parameters?
- Is profiling performed in fp32, or has a validated fp32-shadow alternative
  passed perturbation survival and fidelity?
- Are tokenizer masking, answer positions, PAD/EOS behavior, and truncation
  tested?
- Is `theta0` construction explicit, reproducible, and independent of sealed
  outcomes?
- Are `D_cal`, `D_pred`, `D_prot`, and target rosters mutually disjoint?
- Were neutral, guards, utility, native audit, eligibility, and random streams
  frozen before scoring?
- Did `(R, eta)` pass exact-energy fidelity on this model/block/dtype?
- Were `alpha_pred` and `alpha_prot` selected on different target-free folds?
- Does repair implement Eq. (7), active Eq. (8), hard margins, retries, and
  `B_tok` rather than the old drift guard?
- Can every target row be reproduced from one immutable manifest and clean git
  commit?
- Does the evidence layer expose three separate RQ decisions and fail closed on
  missing support?

If any answer is “no”, keep the run diagnostic and do not merge it into the
claim-bearing tables.

## Current repository implementation boundary

`src/rsus/repair.py` and the separate PDF-v4 wrapper in
`src/rsus/generators/repaired.py` implement the Equation (7)--(8) mechanics,
hard guards, retries, and token metering described above. The old
`src/rsus/stage2.py` remains a legacy diagnostic and must never receive v4
parameter names by translation. `experiments/local_pdf_v4.py` is the local
orchestrator that connects the frozen manifest, two profile channels, exact
Top-Kp allocation, first-reaching parent, and this repair core.

The paper campaign must remain blocked until all of the following are frozen
and the new wrapper is wired into a `PAPER_STAGE_CONTRACT` executor:

- the KL reduction and per-constraint reduction adopted by the paper;
- `beta`, both `kappa` values, `m_ref`, `lambda`, both tolerances,
  `J_retry`, `B_tok`, and the save schedule;
- the semantic exclusion/repair-eligibility manifest;
- dataset-native metrics, orientations, and NI margins; and
- the complete RQ1, RQ2, and RQ3 evidence schema.
