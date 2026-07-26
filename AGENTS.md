# Agent operating contract

This is the only repository-wide instruction file for coding and experiment
agents. `CLAUDE.md` is a pointer to this file. User-facing documentation starts
at `docs/README.md`.

## 1. Identify the task and machine

Before running anything:

```bash
git status --short
git rev-parse --show-toplevel
python experiments/paper/preflight.py
```

Choose one supported path:

| Task | Entry point | Runbook |
|---|---|---|
| RTX 4090 x2, TOFU 1.5B development | `GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh` | `local_run/README.md` |
| Local single-arm diagnostic | `bash local_run.sh <action>` | `local_run/README.md` |
| H100 7B diagnostic campaign | `bash experiments/cluster/run_tofu_7b_h100.sh` | `docs/CLUSTER_FLEET_RUNBOOK.md` |
| H100 14B diagnostic campaign | `bash experiments/cluster/run_tofu_14b_h100.sh` | `docs/CLUSTER_FLEET_RUNBOOK.md` |
| Paper evidence and LaTeX | `experiments/paper/` | `docs/FINAL_RESULTS_RUNBOOK.md` |

The H100 7B/14B scripts run the older channel-matrix campaign. They are not the
latest PDF-v4 complete Table 1/2 workflow. Never relabel their output.

## 2. Source of truth

Use this precedence:

1. The latest paper PDF and frozen paper configs
2. `configs/paper/{campaign,evidence,tofu_v4}.yaml`
3. Freeze files and `prereg/`
4. Current code and tests
5. Current runbooks indexed by `docs/README.md`

Do not use deleted plans, old prompts, chat history, or Git history as current
execution instructions. If a document disagrees with code/config, stop the
launch, verify the contract, and update the existing runbook.

## 3. Scientific invariants

The paper claims prospective selection and fixed denominators. Therefore:

- Select parent hyperparameters only on `D_cal`.
- Select predictor values only on `D_pred`.
- Select protection values only on `D_prot`.
- Never tune, add seeds, or extend a grid after viewing target/audit outcomes.
- Use the first saved checkpoint that reaches the direct-forgetting gate.
- Compare repair arms from the same parent checkpoint, candidate support,
  token/example budget, seed, and guard contract.
- Keep non-reaching, infeasible, and incomplete planned rows in the
  denominator.
- Treat a failed guard as a result to investigate, not a condition to bypass.

Never edit an existing seal, `DONE` marker, run manifest, frozen block, or run
artifact to make a command pass. Never silently replace an unavailable model,
dataset, dtype, metric, or comparator.

## 4. Filesystem and environment

### RTX 4090 workstation

- Reuse `<repo>/.venv`; do not replace a working environment.
- Default model/cache/results paths are documented in `local_run/README.md`.
- Use environment variables for machine-local path overrides.
- Keep generated results out of tracked source directories.

### H100 cluster

- Environment: `/group-volume/fdmu/.venv`
- Shared state/results: `/group-volume/fdmu/runs`
- Host scratch/cache: `/group-volume/fdmu/runtime/<user>/<host>`
- Models: `/group-volume/models`
- Existing dataset source cache: `/group-volume/data/hf_home`

The cluster has no GitHub egress. Do not commit or push there. Do not create a
replacement venv or install packages into the shared environment. Use
`.cluster_env.local.sh` only for supported machine-local path overrides.

Never write cluster results to user-volume, a home filesystem, `/tmp`, or a
shared directory you do not own. `cluster_env.sh` redirects `HOME`, temp files,
and library caches to the runtime path and rejects paths outside
`/group-volume/fdmu`.

## 5. Safe execution

Runtime 장애, 장기 실행 ETA, stale claim, 중간 결과를 조사할 때는 먼저
`docs/LLM_RUN_DIAGNOSTICS.md`를 읽고 그 경로를 직접 검사한다. Terminal 또는
filesystem 접근이 있으면 사용자에게 진단 명령 실행을 떠넘기지 않는다.

Before GPU work:

```bash
python -m pytest -q
nvidia-smi
```

On the cluster also run:

```bash
python experiments/cluster/next_actions.py
```

Only enqueue a phase listed as allowed. A dirty worktree blocks sealed audit
by design. Do not weaken that check; commit/push from a networked workstation,
then pull onto the cluster before workers start.

Existing results are append-only:

- Resume only through a runner's validated resume mechanism.
- Move partial output to `runs/forensics/<name>.<timestamp>` before retrying.
- Never delete a partial run just to reuse its tag.
- Verify a worker is dead on its owning host before `requeue-stale`.
- Read the unit log and classify the root cause before `retry-failed`.

## 6. Result completion

Implementation completion requires tests and syntax checks. Experiment
completion additionally requires all planned units, seals, raw shards, ledger,
readiness report, and generated LaTeX.

Do not report "Table 1/2 complete" unless:

```bash
python experiments/paper/build_evidence.py \
  --config configs/paper/evidence.yaml \
  --ledger results/paper/evidence_ledger.json \
  --paper-root paper \
  --require-ready
```

exits successfully for the intended frozen roster. Placeholder cells and
`all_tables_ready: false` mean incomplete evidence.

## 7. Documentation discipline

- Update an existing canonical document instead of creating a dated plan.
- Keep `docs/README.md` as the complete documentation index.
- Keep commands beside the environment that runs them: local commands in
  `local_run/README.md`, cluster commands in the cluster runbook.
- Do not commit generated Markdown reports or session journals.
- Check relative Markdown links after renaming or deleting files.

## 8. Coding and Git

- Preserve unrelated user changes in a dirty worktree.
- Keep edits scoped and add tests proportional to behavior changed.
- Run focused tests first, then the full CPU suite when practical.
- Run `git diff --check` and shell syntax checks for modified scripts.
- On a networked development machine, commit only verified changes and push
  with a normal fast-forward update. Never force-push unless explicitly asked.
