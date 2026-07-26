#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_ID=qwen25_7b
AUDIT_MATCH="aud__${MODEL_ID}"
STORAGE_ROOT=/group-volume/fdmu
VENV="$STORAGE_ROOT/.venv"
PYTHON="$VENV/bin/python"
CLUSTER_RUNS_ROOT="$STORAGE_ROOT/runs"
# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/user_scope.sh"
QUEUE="$CLUSTER_USER_QUEUE_ROOT/wave2"
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

QUEUE="$CLUSTER_USER_QUEUE_ROOT/wave2"
CURRENT_COMMIT=
printf '[INFO] time=%s stage=environment-bootstrap complete\n' "$(date -u '+%FT%TZ')"

STAGE=bootstrap

print_context() {
  local workers
  workers="$(pgrep -af "experiments/cluster/worker.py --queue" || true)"
  printf '[CONTEXT] host=%s model=%s queue=%s commit=%s python=%s\n' \
    "$(hostname)" "$MODEL_ID" "$QUEUE" "$(git rev-parse --short HEAD)" "$PYTHON"
  printf '[CONTEXT] user=%s runs=%s runtime=%s hf=%s log=%s\n' \
    "$CLUSTER_RUN_USER" "$FDMU_CAMPAIGN_RUNS_ROOT" \
    "$CLUSTER_WORK_ROOT" "$HF_HOME" "$LOG"
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    printf '[CONTEXT] worktree=dirty\n'
    git status --short --untracked-files=all
  else
    printf '[CONTEXT] worktree=clean\n'
  fi
  if [[ -n "$workers" ]]; then
    printf '[CONTEXT] local_workers:\n%s\n' "$workers"
  else
    printf '[CONTEXT] local_workers=none\n'
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  df -h "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" 2>&1 || true
}

on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=%s exit=%s line=%s command=%s\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  case "$STAGE" in
    worker-launch)
      printf '[ANALYSIS] worker launch failed: inspect local_workers below; '\
'an active worker for another queue must finish or be stopped on this host.\n'
      ;;
    enqueue)
      printf '[ANALYSIS] enqueue failed: check worktree state and frozen audit configs below.\n'
      ;;
    *)
      printf '[ANALYSIS] stage command failed; queue and host summaries follow.\n'
      ;;
  esac
  print_context
  "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE" 2>&1 || true
  printf '[ERROR] launcher log retained at %s\n' "$LOG"
  printf '[GUIDE] %s/docs/LLM_RUN_DIAGNOSTICS.md\n' "$ROOT"
  exit "$code"
}
trap on_error ERR

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

print_context
printf '[TOPOLOGY] launcher activates this host only; 7B uses one worker per free local GPU\n'
printf '[TOPOLOGY] run_user=%s queue=%s results=%s\n' \
  "$CLUSTER_RUN_USER" "$QUEUE" "$FDMU_CAMPAIGN_RUNS_ROOT"
printf '[TOPOLOGY] audit_monitor=%s worker_scope=%s; alpha jobs continue independently\n' \
  "$AUDIT_MATCH" "$MODEL_ID"

stage retry-commit-validation
assert_clean_retry_commit

stage failed-audit-partial-quarantine
"$PYTHON" experiments/cluster/quarantine_failed_audit.py \
  --queue "$QUEUE" \
  --config configs/channel_matrix/7b_tofu.yaml \
  --model-id "$MODEL_ID"

stage failed-audit-recovery
"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --queue "$QUEUE" \
  --code-commit "$CURRENT_COMMIT" \
  --unit aud__qwen25_7b__a181 \
  --unit aud__qwen25_7b__a186 \
  --unit aud__qwen25_7b__a191

stage model-queue-commit-reconciliation
"$PYTHON" experiments/cluster/reconcile_queue_commit.py \
  --queue "$QUEUE" \
  --model-id "$MODEL_ID" \
  --code-commit "$CURRENT_COMMIT"

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-7b
stage worker-launch
# Audit units are preferred for early LaTeX. These model-scoped workers then
# continue any queued alpha units and exit naturally without touching other queues.
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
CONFIG=configs/channel_matrix/7b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
SCALE_LABEL=7B \
  bash experiments/channel_matrix/h100_campaign.sh aggregate
printf '[RESULT] LaTeX generation complete: %s\n' \
  "$FDMU_CAMPAIGN_RUNS_ROOT/channel_matrix_7b/aggregate/paper_v4/table1_core_evidence_${MODEL_ID}.tex"
printf '[RESULT] Table 2 LaTeX: %s\n' \
  "$FDMU_CAMPAIGN_RUNS_ROOT/channel_matrix_7b/aggregate/paper_v4/table2_robustness_${MODEL_ID}.tex"
printf '[RESULT] Legacy channel matrix (diagnostic only): %s\n' \
  "$FDMU_CAMPAIGN_RUNS_ROOT/channel_matrix_7b/aggregate/table1_channel_matrix_${MODEL_ID}.tex"
