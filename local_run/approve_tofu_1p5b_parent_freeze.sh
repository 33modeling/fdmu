#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$RUN_ROOT/parent_calibration}"
PROPOSAL="$CALIBRATION_ROOT/freeze_proposals/tofu_parent_freeze_1p5b.recommended.yaml"
SELECTION_INPUT="$CALIBRATION_ROOT/stage/parent_selection_inputs.jsonl"
APPROVAL_LOG="$CALIBRATION_ROOT/freeze_proposals/freeze_validation.log"
PARENT_FREEZE="${PARENT_FREEZE:-$CALIBRATION_ROOT/freeze/tofu_parent_freeze_1p5b.yaml}"
CAMPAIGN="$CALIBRATION_ROOT/config/campaign.local.yaml"
RUNTIME="$CALIBRATION_ROOT/config/tofu_v4.local.yaml"

if [[ ! -x "$PYTHON" ]]; then
  printf 'venv Python is missing; run calibration first: %s\n' "$PYTHON" >&2
  exit 2
fi

bash local_run/ensure_4090_yaml.sh

mkdir -p "$(dirname "$APPROVAL_LOG")"
exec > >(tee -a "$APPROVAL_LOG") 2>&1
printf '[%s] automatic parent freeze validation started\n' "$(date -u '+%FT%TZ')"

for required in "$PROPOSAL" "$SELECTION_INPUT" "$CAMPAIGN" "$RUNTIME"; do
  if [[ ! -f "$required" ]]; then
    printf '[ERROR] parent freeze input is missing: %s\n' "$required" >&2
    exit 2
  fi
done

printf '%s\n' '----- PARENT PROPOSAL BEGIN -----'
cat "$PROPOSAL"
printf '%s\n' '----- PARENT PROPOSAL END -----'

mkdir -p "$(dirname "$PARENT_FREEZE")"
exec "$PYTHON" -u experiments/paper/approve_parent_freeze.py \
  --approve \
  --proposal "$PROPOSAL" \
  --input "$SELECTION_INPUT" \
  --campaign "$CAMPAIGN" \
  --runtime "$RUNTIME" \
  --out "$PARENT_FREEZE"
