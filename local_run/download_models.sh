#!/usr/bin/env bash
# Download the missing <=4B candidates (1.5B essential, 4B nice-to-have) into
# ~/rdata/models. 3B and 7B were copied from /data1/minsoo3.kim already.
set -euo pipefail

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=/rdata/minsoo3.kim/hf_home

MODELS_DIR=/rdata/models
cd /home/minsoo3.kim/dev/retain-susceptibility
source .venv/bin/activate

dl() {  # repo_id  local_subdir
  local repo="$1" dst="$MODELS_DIR/$2"
  if [ -f "$dst/config.json" ] && ls "$dst"/*.safetensors >/dev/null 2>&1; then
    echo "[skip] $2 already present"; return 0
  fi
  echo "[dl] $repo -> $dst"
  hf download "$repo" \
    --local-dir "$dst" \
    --exclude "*.pth" "*.gguf" "original/*" "*.bin" \
    2>&1 | tail -3
}

dl "Qwen/Qwen2.5-1.5B-Instruct" "Qwen2.5-1.5B-Instruct"
dl "Qwen/Qwen3-4B-Instruct-2507" "Qwen3-4B-Instruct-2507"

echo "=== models present ==="
du -sh "$MODELS_DIR"/*/ 2>/dev/null
