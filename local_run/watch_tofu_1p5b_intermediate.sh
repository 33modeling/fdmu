#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b}"
JOINT_ROOT="${JOINT_ROOT:-${RESULTS_ROOT:-$RUN_ROOT/joint_sweep}}"
SPEC="${SPEC:-$ROOT/configs/local/joint_sweep_1p5b_4090x2.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
INTERVAL_SECONDS="${INTERMEDIATE_INTERVAL_SECONDS:-30}"
PARENT_PID=
ONCE=0

while (( $# > 0 )); do
  case "$1" in
    --once) ONCE=1 ;;
    --parent-pid)
      shift
      PARENT_PID="${1:?--parent-pid requires a PID}"
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
  shift
done

[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 2; }
mkdir -p "$JOINT_ROOT/live"
exec 9>"$JOINT_ROOT/live/watcher.lock"
if ! flock -n 9; then
  printf '[live] watcher already active: %s\n' "$JOINT_ROOT/live/watcher.lock"
  exit 0
fi
printf 'pid=%s host=%s started_utc=%s\n' \
  "$$" "$(hostname)" "$(date -u '+%FT%TZ')" > "$JOINT_ROOT/live/watcher.owner"

snapshot() {
  "$PYTHON" experiments/paper/summarize_joint_sweep_live.py \
    --joint-root "$JOINT_ROOT" \
    --spec "$SPEC"
}

while true; do
  snapshot || printf '[live][ERROR] snapshot failed at %s; retrying\n' \
    "$(date -u '+%FT%TZ')" >&2
  (( ONCE == 0 )) || break
  if [[ -n "$PARENT_PID" ]] && ! kill -0 "$PARENT_PID" 2>/dev/null; then
    snapshot || true
    break
  fi
  sleep "$INTERVAL_SECONDS"
done

printf '[live] final snapshot: %s/live/LIVE_STATUS.md\n' "$JOINT_ROOT"
