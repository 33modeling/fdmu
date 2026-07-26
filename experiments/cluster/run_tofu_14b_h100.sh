#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_ID=qwen25_14b
AUDIT_MATCH="aud__${MODEL_ID}"
STORAGE_ROOT=/group-volume/fdmu
VENV="$STORAGE_ROOT/.venv"
PYTHON="$VENV/bin/python"
QUEUE="$STORAGE_ROOT/runs/cluster_queue/wave1_14b"
LOG_DIR="$STORAGE_ROOT/runs/logs/cluster"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_$(date -u '+%Y%m%dT%H%M%SZ').out"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_current.out"
exec > >(tee -a "$LOG") 2>&1

bootstrap_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=environment-bootstrap exit=%s line=%s command=%s\n' \
    "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  df -h "$STORAGE_ROOT" 2>&1 || true
  printf '[ERROR] launcher log retained at %s\n' "$LOG"
  exit "$code"
}
trap bootstrap_error ERR

cluster_environment_imports_ready() {
  [[ -x "$PYTHON" ]] \
    && "$PYTHON" -c 'import datasets, torch, transformers, yaml' >/dev/null 2>&1
}

printf '[INFO] time=%s stage=environment-bootstrap start\n' "$(date -u '+%FT%TZ')"
if [[ "${BOOTSTRAP_CLUSTER_ENV:-1}" == "1" ]]; then
  if cluster_environment_imports_ready; then
    printf '[INFO] existing cluster environment imports pass; skipping shared setup lock: %s\n' \
      "$VENV"
  else
    bash "$ROOT/experiments/cluster/setup_group_volume.sh"
  fi
fi

# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave1_14b"
WORKER_GPU=0
FIDELITY_GPU="${FIDELITY_GPU:-0}"
CURRENT_COMMIT=
printf '[INFO] time=%s stage=environment-bootstrap complete\n' "$(date -u '+%FT%TZ')"

STAGE=bootstrap
on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=%s exit=%s line=%s command=%s\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  printf '[CONTEXT] host=%s model=%s queue=%s commit=%s python=%s\n' \
    "$(hostname)" "$MODEL_ID" "$QUEUE" "$(git rev-parse --short HEAD)" "$PYTHON"
  printf '[CONTEXT] runs=%s runtime=%s hf=%s log=%s\n' \
    "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" "$HF_HOME" "$LOG"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  df -h "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" 2>&1 || true
  "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE" 2>&1 || true
  printf '[ERROR] full launcher log: %s\n' "$LOG"
  exit "$code"
}
trap on_error ERR

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] missing environment: $VENV/bin/activate" >&2
  exit 1
fi
# Queue units invoke `python`; activate the pinned environment so they resolve
# to /group-volume/fdmu/.venv/bin/python.
# shellcheck disable=SC1090
source "$VENV/bin/activate"

stage() {
  STAGE="$1"
  printf '[INFO] time=%s stage=%s start\n' "$(date -u '+%FT%TZ')" "$STAGE"
}

assert_clean_retry_commit() {
  local tree_status
  tree_status="$(git status --porcelain --untracked-files=all)"
  if [[ -n "$tree_status" ]]; then
    printf '[ERROR] refusing queue re-pin from a dirty worktree:\n%s\n' \
      "$tree_status" >&2
    return 2
  fi
  CURRENT_COMMIT="$(git rev-parse HEAD)"
  printf '[INFO] retry units will be pinned to clean commit=%s\n' \
    "$CURRENT_COMMIT"
}

assert_14b_gpu_exclusive() {
  local process gpu
  local -a extra_wave_workers=()
  local -a gpu0_workers=()

  if [[ "$FIDELITY_GPU" != "0" ]]; then
    echo "[ERROR] 14B fidelity and worker are pinned to GPU 0; FIDELITY_GPU=$FIDELITY_GPU" >&2
    return 2
  fi
  if ! command -v nvidia-smi >/dev/null; then
    echo "[ERROR] nvidia-smi not found; cannot verify the dedicated 14B GPU" >&2
    return 2
  fi

  while IFS= read -r process; do
    [[ -n "$process" ]] || continue
    if [[ ! "$process" =~ --queue[[:space:]]+([^[:space:]]*/)?cluster_queue/wave1_14b([[:space:]]|$) ]]; then
      continue
    fi
    if [[ "$process" =~ --gpu[[:space:]]+([0-9]+) ]]; then
      gpu="${BASH_REMATCH[1]}"
      if (( gpu > 0 )); then
        extra_wave_workers+=("$process")
      fi
    else
      extra_wave_workers+=("$process")
    fi
  done < <(pgrep -af "experiments/cluster/worker.py --queue" || true)
  if (( ${#extra_wave_workers[@]} > 0 )); then
    printf '[ERROR] wave1_14b must not retain GPU1-7 workers:\n' >&2
    printf '  %s\n' "${extra_wave_workers[@]}" >&2
    echo "[ERROR] stop those workers and let their claimed units settle before restarting" >&2
    return 2
  fi

  while IFS= read -r process; do
    [[ -n "$process" ]] || continue
    if [[ "$process" =~ --gpu[[:space:]]+0([[:space:]]|$) ]]; then
      gpu0_workers+=("$process")
    fi
  done < <(pgrep -af "experiments/cluster/worker.py --queue" || true)
  if (( ${#gpu0_workers[@]} > 0 )); then
    printf '[ERROR] GPU 0 is reserved by an existing cluster worker; fidelity not started:\n' >&2
    printf '  %s\n' "${gpu0_workers[@]}" >&2
    return 2
  fi

  local gpu_processes
  if ! gpu_processes="$(
    nvidia-smi -i 0 \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits
  )"; then
    echo "[ERROR] failed to inspect GPU 0 compute processes" >&2
    return 2
  fi
  if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
    printf '[ERROR] GPU 0 already has active compute processes; fidelity not started:\n%s\n' \
      "$gpu_processes" >&2
    return 2
  fi
  printf '[INFO] GPU 0 exclusive preflight passed for wave1_14b\n'
}

stage gpu-exclusive-preflight
printf '[TOPOLOGY] launcher activates this host only; 14B uses one dedicated worker on GPU 0\n'
printf '[TOPOLOGY] audit_monitor=%s worker_scope=%s; alpha jobs continue independently\n' \
  "$AUDIT_MATCH" "$MODEL_ID"
assert_14b_gpu_exclusive

stage retry-commit-validation
assert_clean_retry_commit

stage fidelity
GPU="$FIDELITY_GPU" \
CONFIG=configs/channel_matrix/14b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

stage failed-audit-partial-quarantine
"$PYTHON" experiments/cluster/quarantine_failed_audit.py \
  --queue "$QUEUE" \
  --config configs/channel_matrix/14b_tofu.yaml \
  --model-id "$MODEL_ID"

stage failed-audit-recovery
AUDIT_IDS_RAW="$(
  "$PYTHON" - <<'PY'
from pathlib import Path

import yaml

config = yaml.safe_load(
    Path("configs/channel_matrix/14b_tofu.yaml").read_text(encoding="utf-8")
)
for author in config["audit"]["authors"]:
    print(f"aud__qwen25_14b__a{int(author)}")
PY
)"
mapfile -t AUDIT_UNIT_IDS <<<"$AUDIT_IDS_RAW"
RETRY_ARGS=()
for unit_id in "${AUDIT_UNIT_IDS[@]}"; do
  [[ -n "$unit_id" ]] || continue
  RETRY_ARGS+=(--unit "$unit_id")
done
if (( ${#RETRY_ARGS[@]} == 0 )); then
  echo "[ERROR] no qwen25_14b audit unit IDs resolved from config" >&2
  exit 2
fi
printf '[INFO] retry scope: %s\n' "${AUDIT_UNIT_IDS[*]}"
"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --code-commit "$CURRENT_COMMIT" \
  --queue "$QUEUE" "${RETRY_ARGS[@]}"

stage fidelity-contract-validation
"$PYTHON" experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/14b_tofu.yaml --phase audit \
  --model-id "$MODEL_ID" --only-authors 181 \
  --dry-run --limit 1

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-14b
stage worker-launch
printf '[CONFIG] model=%s worker_gpu=%s queue=%s python=%s\n' \
  "$MODEL_ID" "$WORKER_GPU" "$QUEUE" "$PYTHON"
# Audit units are preferred for early LaTeX. The GPU 0 worker then continues
# queued alpha units and exits naturally without touching other queues.
WAIT=0 UNIT_MATCH="$MODEL_ID" UNIT_PREFER="$AUDIT_MATCH" \
  bash experiments/cluster/launch_node.sh --dedicated-queue "$QUEUE" 1
printf '[RESULT] worker-launch complete; active local workers:\n'
pgrep -af "experiments/cluster/worker.py --queue" || true
"$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE"
stage queue-monitor
"$PYTHON" -u experiments/cluster/monitor_queue.py \
  --queue "$QUEUE" --match "$AUDIT_MATCH" \
  --interval "${MONITOR_INTERVAL_SECONDS:-30}"
stage aggregate-latex
CONFIG=configs/channel_matrix/14b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
SCALE_LABEL=14B \
  bash experiments/channel_matrix/h100_campaign.sh aggregate
printf '[RESULT] LaTeX generation complete: %s\n' \
  "$CLUSTER_RUNS_ROOT/channel_matrix_14b/aggregate/table1_channel_matrix_${MODEL_ID}.tex"
