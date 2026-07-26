#!/usr/bin/env bash
set -Eeuo pipefail

# Attach three additional workers to a 14B run that was started by the former
# single-worker launcher. This preserves the in-progress fidelity computation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STORAGE_ROOT=/group-volume/fdmu
VENV="$STORAGE_ROOT/.venv"
PYTHON="$VENV/bin/python"
MODEL_ID=qwen25_14b
AUDIT_MATCH="aud__${MODEL_ID}"
AUDIT_GPU_COUNT="${AUDIT_GPU_COUNT:-4}"

# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave1_14b"
LAUNCHER_LOG="$CLUSTER_RUNS_ROOT/logs/cluster/launcher_${MODEL_ID}_$(hostname)_current.out"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
started="$(date +%s)"

printf '[SCALE] waiting for existing 14B launcher worker stage: %s\n' "$LAUNCHER_LOG"
while ! grep -q 'stage=worker-launch start' "$LAUNCHER_LOG" 2>/dev/null; do
  if grep -q '^\[ERROR\] stage=' "$LAUNCHER_LOG" 2>/dev/null; then
    echo "[ERROR] existing 14B launcher failed before worker launch: $LAUNCHER_LOG" >&2
    tail -n 80 "$LAUNCHER_LOG" >&2 || true
    exit 2
  fi
  elapsed="$(( $(date +%s) - started ))"
  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "[ERROR] timed out after ${elapsed}s waiting for 14B worker stage" >&2
    exit 2
  fi
  printf '[SCALE] fidelity/queue preparation still running elapsed_s=%s\n' "$elapsed"
  sleep "${WAIT_INTERVAL_SECONDS:-30}"
done

"$PYTHON" - "$ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "experiments" / "channel_matrix"))
import run_campaign

config = run_campaign._load_yaml(root / "configs/channel_matrix/14b_tofu.yaml")
model = run_campaign._enabled_models(config, {"qwen25_14b"})[0]
validated = run_campaign.validate_fidelity_artifact_pair(config, model)
print(f"[SCALE] validated fidelity certificate: {validated['path']}", flush=True)
PY

printf '[SCALE] attaching up to %s total 14B workers without restarting completed work\n' \
  "$AUDIT_GPU_COUNT"
WAIT=0 UNIT_MATCH="$MODEL_ID" UNIT_PREFER="$AUDIT_MATCH" \
  bash experiments/cluster/launch_node.sh \
    --dedicated-queue "$QUEUE" "$AUDIT_GPU_COUNT"
