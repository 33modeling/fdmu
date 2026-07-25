#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

QUEUE="${H100_14B_QUEUE:-runs/cluster_queue/wave1_14b}"
bash experiments/cluster/enqueue_table12.sh audit-14b
FORCE_QUEUE="${FORCE_QUEUE:-1}" \
  bash experiments/cluster/launch_node.sh "$QUEUE"
