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

APPROVAL_RECORD="$CALIBRATION_ROOT/freeze_proposals/parent_freeze_approval.json"
if "$PYTHON" - "$PARENT_FREEZE" "$PROPOSAL" "$SELECTION_INPUT" "$APPROVAL_RECORD" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import yaml

freeze_path, proposal_path, selection_path, record_path = (
    Path(value).resolve() for value in sys.argv[1:]
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    valid = (
        isinstance(freeze, dict)
        and freeze.get("status") == "frozen"
        and record.get("schema_version") == 1
        and record.get("approved") is True
        and Path(str(record.get("output", ""))).resolve() == freeze_path
        and record.get("output_sha256") == digest(freeze_path)
        and Path(str(record.get("proposal", ""))).resolve() == proposal_path
        and record.get("proposal_sha256") == digest(proposal_path)
        and Path(str(record.get("selection_input", ""))).resolve()
        == selection_path
        and record.get("selection_input_sha256") == digest(selection_path)
    )
except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
    print(f"[PARENT_FREEZE_STATUS] complete=false reason={type(error).__name__}: {error}")
    raise SystemExit(1)
if not valid:
    print("[PARENT_FREEZE_STATUS] complete=false reason=approval record mismatch")
    raise SystemExit(1)
print(f"[PARENT_FREEZE_STATUS] complete=true freeze={freeze_path}")
PY
then
  printf '[PARENT FREEZE SKIPPED] approved freeze is complete; recompute=0\n'
  exit 0
fi

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
