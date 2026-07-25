#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

QUEUE="${H100_7B_QUEUE:-runs/cluster_queue/wave2}"
bash experiments/cluster/enqueue_table12.sh audit-7b
FORCE_QUEUE="${FORCE_QUEUE:-1}" \
  bash experiments/cluster/launch_node.sh "$QUEUE"
