#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/parent_calibration}"
PROPOSAL="$CALIBRATION_ROOT/freeze_proposals/tofu_parent_freeze_1p5b.recommended.yaml"
SELECTION_INPUT="$CALIBRATION_ROOT/stage/parent_selection_inputs.jsonl"
APPROVAL_LOG="$CALIBRATION_ROOT/freeze_proposals/approval.log"

if [[ ! -x "$PYTHON" ]]; then
  printf 'venv Python is missing; run calibration first: %s\n' "$PYTHON" >&2
  exit 2
fi

mkdir -p "$(dirname "$APPROVAL_LOG")"
exec > >(tee -a "$APPROVAL_LOG") 2>&1
printf '[%s] parent freeze approval requested\n' "$(date -u '+%FT%TZ')"

exec "$PYTHON" -u experiments/paper/approve_parent_freeze.py \
  --proposal "$PROPOSAL" \
  --input "$SELECTION_INPUT" \
  "$@"
