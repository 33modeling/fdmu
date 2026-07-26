#!/usr/bin/env bash
set -Eeuo pipefail

# CPU-only finalization of existing 7B results. This entry point never mutates
# a queue and never starts, retries, or stops a GPU worker.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

printf '[RENDER-ONLY] existing 7B runs will be read; GPU work will not be launched\n'
printf '[RENDER-ONLY] action=paper-v4 config=configs/channel_matrix/7b_tofu.yaml model=qwen25_7b\n'

exec env \
  CONFIG=configs/channel_matrix/7b_tofu.yaml \
  MODEL_ID=qwen25_7b \
  SCALE_LABEL=7B \
  bash experiments/channel_matrix/h100_campaign.sh paper-v4
