#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

QUEUE="${H100_7B_QUEUE:-runs/cluster_queue/wave2}"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
PYTHON="$VENV/bin/python"

GPU="${FIDELITY_GPU:-0}" \
CONFIG=configs/channel_matrix/7b_tofu.yaml \
MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

"$PYTHON" experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase audit --model-id qwen25_7b --only-authors 181 \
  --dry-run --limit 1

"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --queue "$QUEUE" \
  --unit aud__qwen25_7b__a181 \
  --unit aud__qwen25_7b__a186 \
  --unit aud__qwen25_7b__a191

bash experiments/cluster/enqueue_table12.sh audit-7b
FORCE_QUEUE="${FORCE_QUEUE:-0}" \
  bash experiments/cluster/launch_node.sh "$QUEUE"
