#!/usr/bin/env bash
set -Eeuo pipefail

# Bring this node into the fleet: start its status watcher and one queue
# worker per GPU. Safe to re-run — GPUs that already have a worker for this
# queue are skipped, and the watcher is single-instance.
#
#   bash experiments/cluster/launch_node.sh              # queue from configs/cluster/fleet.yaml
#   bash experiments/cluster/launch_node.sh <queue-dir>  # explicit (must match assignment)
#   bash experiments/cluster/launch_node.sh --dedicated-queue <queue-dir>
#                                      # model launcher pins a dedicated host
#   WAIT=0 ...                                           # workers exit when queue drains
#
# Stop this node's workers:  pkill -f "experiments/cluster/worker.py --queue"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/group-volume/fdmu/.venv"
PYTHON="$VENV/bin/python"
WAIT="${WAIT:-1}"
UNIT_MATCH="${UNIT_MATCH:-}"
UNIT_PREFER="${UNIT_PREFER:-}"
HOST="$(hostname)"
DEDICATED_QUEUE=0
if [[ "${1:-}" == "--dedicated-queue" ]]; then
  DEDICATED_QUEUE=1
  shift
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing official environment: ${VENV}/bin/activate" >&2
  echo "run: bash experiments/cluster/setup_group_volume.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
if [[ ! -x "$PYTHON" ]]; then
  echo "official Python is missing or not executable: $PYTHON" >&2
  exit 1
fi
export FDMU_WORKER_PYTHON="$PYTHON"
"$PYTHON" -c 'import sys, yaml; print(
    f"[worker-python] executable={sys.executable} prefix={sys.prefix} yaml={yaml.__file__}"
)'
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/experiments/cluster/cluster_env.sh"
export PYTHONUNBUFFERED=1

ASSIGNED="$("$PYTHON" - <<PY
import yaml, pathlib
cfg = pathlib.Path("configs/cluster/fleet.yaml")
data = yaml.safe_load(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
print((data.get("assignments") or {}).get("${HOST}", ""))
PY
)"
if [[ "${ASSIGNED}" == runs/cluster_queue/users/* ]]; then
  ASSIGNED="$CLUSTER_RUNS_ROOT/${ASSIGNED#runs/}"
elif [[ "${ASSIGNED}" == runs/cluster_queue/* ]]; then
  ASSIGNED="$CLUSTER_USER_QUEUE_ROOT/${ASSIGNED#runs/cluster_queue/}"
elif [[ "${ASSIGNED}" == runs/* ]]; then
  ASSIGNED="$CLUSTER_RUNS_ROOT/${ASSIGNED#runs/}"
fi

QUEUE="${1:-}"
if [[ "${QUEUE}" == runs/cluster_queue/users/* ]]; then
  QUEUE="$CLUSTER_RUNS_ROOT/${QUEUE#runs/}"
elif [[ "${QUEUE}" == runs/cluster_queue/* ]]; then
  QUEUE="$CLUSTER_USER_QUEUE_ROOT/${QUEUE#runs/cluster_queue/}"
elif [[ "${QUEUE}" == runs/* ]]; then
  QUEUE="$CLUSTER_RUNS_ROOT/${QUEUE#runs/}"
fi
if (( DEDICATED_QUEUE == 1 )); then
  if [[ -z "${QUEUE}" ]]; then
    echo "--dedicated-queue requires an explicit queue directory" >&2
    exit 2
  fi
  case "${QUEUE%/}/" in
    "${CLUSTER_USER_QUEUE_ROOT%/}/"*) ;;
    *)
      echo "dedicated queue must be under ${CLUSTER_USER_QUEUE_ROOT}: ${QUEUE}" >&2
      exit 2
      ;;
  esac
  echo "node=${HOST} user=${CLUSTER_RUN_USER} mode=dedicated queue=${QUEUE}; fleet hostname assignment not required"
else
  if [[ -z "${ASSIGNED}" ]]; then
    echo "node ${HOST} has no assignment in configs/cluster/fleet.yaml" >&2
    echo "Add and commit the exact queue assignment before launching this node." >&2
    exit 2
  fi
  if [[ -z "${QUEUE}" ]]; then
    QUEUE="${ASSIGNED}"
  elif [[ "${QUEUE%/}" != "${ASSIGNED%/}" ]]; then
    echo "refusing: ${HOST} is assigned to ${ASSIGNED}, not ${QUEUE}." >&2
    echo "Fix and commit configs/cluster/fleet.yaml before launching this node." >&2
    exit 2
  fi
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found; this launcher is for GPU nodes" >&2
  exit 1
fi
DETECTED="$(nvidia-smi -L | wc -l)"
NGPU="${2:-${DETECTED}}"
if (( NGPU > DETECTED )); then
  echo "requested ${NGPU} GPUs but node has ${DETECTED}" >&2
  exit 1
fi

# Serialize the conflict check and worker spawn on this host. Without this
# short-lived lease, simultaneous 7B/14B launchers can both observe GPU 0 as
# free and then double-book it.
command -v flock >/dev/null 2>&1 \
  || { echo "flock is required for host launcher coordination" >&2; exit 2; }
HOST_LAUNCH_LOCK="$CLUSTER_WORK_ROOT/launcher.lock"
exec 8>"$HOST_LAUNCH_LOCK"
if ! flock -x -w "${HOST_LAUNCH_LOCK_TIMEOUT_SECONDS:-60}" 8; then
  echo "timed out waiting for host launcher lease: $HOST_LAUNCH_LOCK" >&2
  echo "check for another active 7B/14B launcher on this host" >&2
  exit 2
fi
echo "host launcher lease acquired: $HOST_LAUNCH_LOCK"

# Refuse before launching anything when this node already serves another
# queue. Queue-specific duplicate checks alone can otherwise double-book all
# GPUs and make every newly spawned worker fail independently.
CONFLICTS=()
while IFS= read -r process; do
  [[ -n "${process}" ]] || continue
  if [[ "${process}" == *"experiments/cluster/worker.py --queue ${QUEUE} "* ]]; then
    continue
  fi
  CONFLICTS+=("${process}")
done < <(pgrep -af "experiments/cluster/worker.py --queue" || true)
if (( ${#CONFLICTS[@]} > 0 )); then
  printf 'refusing to double-book this node; workers for another queue are active:\n' >&2
  printf '  %s\n' "${CONFLICTS[@]}" >&2
  printf 'stop or move those workers before launching queue %s\n' "${QUEUE}" >&2
  exit 2
fi

LOGDIR="$CLUSTER_USER_RUNS_ROOT/logs/cluster"
mkdir -p "${LOGDIR}"
"$PYTHON" experiments/cluster/workqueue.py init --queue "${QUEUE}"

nohup "$PYTHON" -u experiments/cluster/node_watch.py --replace \
  --status-dir "$CLUSTER_USER_RUNS_ROOT/cluster_status" \
  8>&- >> "${LOGDIR}/watch_${HOST}.out" 2>&1 &
echo "node=${HOST} queue=${QUEUE} gpus=${NGPU}/${DETECTED} wait=${WAIT} unit_match=${UNIT_MATCH:-all} unit_prefer=${UNIT_PREFER:-none} watcher_pid=$!"

wait_flag=()
if [[ "${WAIT}" == "1" ]]; then
  wait_flag=(--wait)
fi
match_flag=()
if [[ -n "${UNIT_MATCH}" ]]; then
  match_flag=(--match-unit "${UNIT_MATCH}")
fi
prefer_flag=()
if [[ -n "${UNIT_PREFER}" ]]; then
  prefer_flag=(--prefer-unit-prefix "${UNIT_PREFER}")
fi

STARTED_PIDS=()
STARTED_LOGS=()
for (( g = 0; g < NGPU; g++ )); do
  if pgrep -f "experiments/cluster/worker.py --queue ${QUEUE} --gpu ${g}( |$)" >/dev/null; then
    echo "  worker gpu${g}: already running, skipped"
    continue
  fi
  out="${LOGDIR}/worker_${HOST}_gpu${g}.out"
  nohup "$PYTHON" -u experiments/cluster/worker.py \
    --queue "${QUEUE}" --gpu "${g}" "${wait_flag[@]}" \
    "${match_flag[@]}" "${prefer_flag[@]}" \
    --log-dir "$LOGDIR" \
    8>&- >> "${out}" 2>&1 &
  pid=$!
  STARTED_PIDS+=("$pid")
  STARTED_LOGS+=("$out")
  echo "  worker gpu${g}: started pid=${pid} log=${out}"
done

sleep "${WORKER_STARTUP_GRACE_SECONDS:-3}"
STARTUP_FAILURES=0
for index in "${!STARTED_PIDS[@]}"; do
  pid="${STARTED_PIDS[$index]}"
  out="${STARTED_LOGS[$index]}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "  worker startup verified: pid=${pid} log=${out}"
    continue
  fi
  if tail -n 20 "$out" 2>/dev/null | grep -q "queue drained"; then
    echo "  worker exited cleanly after queue drain: pid=${pid} log=${out}"
    continue
  fi
  echo "worker died during startup: pid=${pid} log=${out}" >&2
  tail -n 60 "$out" >&2 || true
  STARTUP_FAILURES=$((STARTUP_FAILURES + 1))
done
if (( STARTUP_FAILURES > 0 )); then
  echo "${STARTUP_FAILURES} worker(s) failed during startup" >&2
  exit 2
fi
echo "overview: bash experiments/cluster/fleet_status.sh"
