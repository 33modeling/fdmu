#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b}"
MODEL_PATH="${MODEL_PATH:-/rdata/models/Qwen2.5-1.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1}"
FIDELITY_GPU="${FIDELITY_GPU:-${GPU_IDS%%,*}}"
FIDELITY_ROOT="${FIDELITY_ROOT:-$RUN_ROOT/fidelity}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="$FIDELITY_ROOT/config/1p5b_tofu.local.yaml"
CAMPAIGN_ROOT="$FIDELITY_ROOT/channel_matrix_1p5b"
CERTIFICATE="$CAMPAIGN_ROOT/fidelity/qwen25_1p5b.json"
SUMMARY="$FIDELITY_ROOT/fidelity_summary.json"
LOG_DIR="$FIDELITY_ROOT/launcher_logs"

mkdir -p "$LOG_DIR" "$(dirname "$CONFIG")"
LOG="$LOG_DIR/$(date -u '+%Y%m%dT%H%M%SZ')-$$.log"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/current.log"
exec > >(tee -a "$LOG") 2>&1

on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] 1.5B fidelity exit=%s line=%s command=%s\n' \
    "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  printf '[ERROR] fidelity log retained at %s\n' "$LOG"
  exit "$code"
}
trap on_error ERR

printf '[%s] declared-fidelity start pid=%s\n' "$(date -u '+%FT%TZ')" "$$"
printf '[CONFIG] model=%s output=%s certificate=%s summary=%s\n' \
  "$MODEL_PATH" "$CAMPAIGN_ROOT" "$CERTIFICATE" "$SUMMARY"
printf '[CONFIG] fidelity_gpu=%s\n' "$FIDELITY_GPU"

if [[ -x "$PYTHON" ]] \
  && "$PYTHON" - "$SUMMARY" "$CERTIFICATE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

summary_path, certificate_path = (Path(value).resolve() for value in sys.argv[1:])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    valid = (
        summary.get("setting") == "tofu_qwen25_1p5b"
        and summary.get("support") == "declared_setting_fidelity"
        and Path(str(summary.get("source_certificate", ""))).resolve()
        == certificate_path
        and summary.get("source_certificate_sha256")
        == hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        and summary.get("certificate_passed") is True
        and certificate.get("passed") is True
    )
except (OSError, ValueError, TypeError) as error:
    print(f"[FIDELITY_STATUS] complete=false reason={type(error).__name__}: {error}")
    raise SystemExit(1)
if not valid:
    print("[FIDELITY_STATUS] complete=false reason=summary/certificate mismatch")
    raise SystemExit(1)
print(f"[FIDELITY_STATUS] complete=true summary={summary_path}")
PY
then
  printf '[DECLARED FIDELITY SKIPPED] validated summary already exists; rerun=0\n'
  exit 0
fi

if [[ "${FDMU_4090_BOOTSTRAPPED:-0}" != "1" ]]; then
  bash local_run/bootstrap_4090_env.sh
fi

"$PYTHON" - "$MODEL_PATH" "$CAMPAIGN_ROOT" "$CERTIFICATE" "$CONFIG" <<'PY'
from pathlib import Path
import sys

import yaml

model_path, output_root, certificate, destination = map(Path, sys.argv[1:])
source = Path("configs/channel_matrix/1p5b_tofu.yaml")
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["output_root"] = str(output_root.resolve())
models = [
    model for model in config["models"] if model.get("id") == "qwen25_1p5b"
]
if len(models) != 1:
    raise SystemExit("1.5B fidelity config must contain exactly qwen25_1p5b")
models[0]["path"] = str(model_path.resolve())
models[0]["enabled"] = True
config["models"] = models
config["common"].pop("sentence_encoder", None)
config["audit"]["predictors"] = [
    predictor
    for predictor in config["audit"]["predictors"]
    if predictor != "knn_embed"
]
config["audit"]["fidelity_certificates"] = {
    "qwen25_1p5b": str(certificate.resolve())
}
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
temporary.replace(destination)
print(f"[CONFIG] wrote runtime fidelity config: {destination}")
PY

export HF_HOME="${HF_HOME:-/rdata/minsoo3.kim/hf_home}"
export RSUS_DATASETS_CACHE="${RSUS_DATASETS_CACHE:-$HF_HOME/rsus_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="$FIDELITY_GPU" \
  "$PYTHON" -u experiments/channel_matrix/run_campaign.py \
    --config "$CONFIG" \
    --phase fidelity \
    --model-id qwen25_1p5b \
    --resume

"$PYTHON" -u experiments/paper/export_channel_matrix_raw.py \
  --campaign-config configs/paper/campaign.yaml \
  --setting-id tofu_qwen25_1p5b \
  --out-dir "$FIDELITY_ROOT/export" \
  --skip-prediction \
  --skip-protection \
  --fidelity-certificate "$CERTIFICATE" \
  --fidelity-out "$SUMMARY"

"$PYTHON" - "$SUMMARY" "$CERTIFICATE" <<'PY'
import json
from pathlib import Path
import sys

summary_path, certificate_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
if summary.get("setting") != "tofu_qwen25_1p5b":
    raise SystemExit("fidelity summary setting mismatch")
if summary.get("support") != "declared_setting_fidelity":
    raise SystemExit("fidelity summary support is not declared_setting_fidelity")
if certificate.get("passed") is not True:
    raise SystemExit("declared fidelity certificate did not pass")
print(f"[RESULT] fidelity_certificate={certificate_path}")
print(f"[RESULT] fidelity_summary={summary_path}")
PY

printf '[%s] declared-fidelity complete log=%s\n' "$(date -u '+%FT%TZ')" "$LOG"
