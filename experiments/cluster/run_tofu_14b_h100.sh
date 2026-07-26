#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_ID=qwen25_14b
AUDIT_MATCH="aud__${MODEL_ID}"
STORAGE_ROOT=/group-volume/fdmu
VENV="$STORAGE_ROOT/.venv"
PYTHON="$VENV/bin/python"
CLUSTER_RUNS_ROOT="$STORAGE_ROOT/runs"
# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/user_scope.sh"
QUEUE="$CLUSTER_USER_QUEUE_ROOT/wave1_14b"
LOG_DIR="$CLUSTER_USER_RUNS_ROOT/logs/cluster"
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

printf '[INFO] time=%s stage=environment-bootstrap start\n' "$(date -u '+%FT%TZ')"
if [[ "${BOOTSTRAP_CLUSTER_ENV:-1}" == "1" ]]; then
  bash "$ROOT/experiments/cluster/setup_group_volume.sh"
fi

# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_USER_QUEUE_ROOT/wave1_14b"
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
  printf '[CONTEXT] user=%s runs=%s runtime=%s hf=%s log=%s\n' \
    "$CLUSTER_RUN_USER" "$FDMU_CAMPAIGN_RUNS_ROOT" \
    "$CLUSTER_WORK_ROOT" "$HF_HOME" "$LOG"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  df -h "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" 2>&1 || true
  "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE" 2>&1 || true
  printf '[ERROR] full launcher log: %s\n' "$LOG"
  printf '[GUIDE] %s/docs/LLM_RUN_DIAGNOSTICS.md\n' "$ROOT"
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

printf '[TOPOLOGY] launcher activates this host only; 14B uses one worker per free local GPU\n'
printf '[TOPOLOGY] run_user=%s queue=%s results=%s\n' \
  "$CLUSTER_RUN_USER" "$QUEUE" "$FDMU_CAMPAIGN_RUNS_ROOT"
printf '[TOPOLOGY] audit_monitor=%s worker_scope=%s; alpha jobs continue independently\n' \
  "$AUDIT_MATCH" "$MODEL_ID"

stage retry-commit-validation
assert_clean_retry_commit

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

stage model-queue-commit-reconciliation
"$PYTHON" experiments/cluster/reconcile_queue_commit.py \
  --queue "$QUEUE" \
  --model-id "$MODEL_ID" \
  --code-commit "$CURRENT_COMMIT"

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-14b
stage worker-launch
printf '[CONFIG] model=%s workers=all-free-local-gpus queue=%s python=%s\n' \
  "$MODEL_ID" "$QUEUE" "$PYTHON"
# Audit units are preferred for early LaTeX. Existing workers for this same
# queue are preserved; launch_node adds one worker on every other free GPU.
WAIT=0 UNIT_MATCH="$MODEL_ID" UNIT_PREFER="$AUDIT_MATCH" \
  bash experiments/cluster/launch_node.sh --dedicated-queue "$QUEUE"
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
  "$FDMU_CAMPAIGN_RUNS_ROOT/channel_matrix_14b/aggregate/table1_channel_matrix_${MODEL_ID}.tex"
