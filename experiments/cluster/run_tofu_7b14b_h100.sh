#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

QUEUE="${H100_7B14B_QUEUE:-runs/cluster_queue/wave_tofu_7b14b}"
ACTION="${1:-run}"

case "$ACTION" in
  run)
    H100_7B14B_QUEUE="$QUEUE" \
      bash experiments/cluster/enqueue_table12.sh audit-7b14b
    FORCE_QUEUE="${FORCE_QUEUE:-1}" \
      bash experiments/cluster/launch_node.sh "$QUEUE"
    ;;
  enqueue)
    H100_7B14B_QUEUE="$QUEUE" \
      bash experiments/cluster/enqueue_table12.sh audit-7b14b
    ;;
  launch)
    FORCE_QUEUE="${FORCE_QUEUE:-1}" \
      bash experiments/cluster/launch_node.sh "$QUEUE"
    ;;
  status)
    source "${VENV:-/group-volume/jieuns.shin/venvs/exp}/bin/activate"
    python experiments/cluster/workqueue.py status --brief --queue "$QUEUE"
    ;;
  *)
    printf 'usage: %s [run|enqueue|launch|status]\n' "$0" >&2
    exit 2
    ;;
esac
