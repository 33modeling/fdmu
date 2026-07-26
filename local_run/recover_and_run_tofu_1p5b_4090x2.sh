#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi is required" >&2
  exit 2
fi

declare -A GPU_PIDS=()
collect_pids() {
  local gpu pid
  GPU_PIDS=()
  for gpu in "${GPUS[@]}"; do
    while IFS= read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ "$pid" =~ ^[0-9]+$ ]] && GPU_PIDS["$pid"]=1
    done < <(
      nvidia-smi --id="$gpu" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )
  done
}

echo "[RECOVERY] selected GPUs: $GPU_IDS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
collect_pids

if (( ${#GPU_PIDS[@]} > 0 )); then
  echo "[RECOVERY] terminating existing compute processes on GPUs $GPU_IDS"
  for pid in "${!GPU_PIDS[@]}"; do
    ps -o user,pid,ppid,stat,cmd -p "$pid" || true
  done
  sudo kill -TERM "${!GPU_PIDS[@]}" 2>/dev/null || true
  sleep 10
  collect_pids
fi

if (( ${#GPU_PIDS[@]} > 0 )); then
  echo "[RECOVERY] force-killing remaining compute processes"
  sudo kill -KILL "${!GPU_PIDS[@]}" 2>/dev/null || true
  sleep 5
  collect_pids
fi

if (( ${#GPU_PIDS[@]} > 0 )); then
  printf '[ERROR] GPU processes survived SIGKILL: %s\n' "${!GPU_PIDS[*]}" >&2
  echo "[ERROR] run sudo reboot, then rerun this script" >&2
  exit 2
fi

for gpu in "${GPUS[@]}"; do
  used="$(
    nvidia-smi --id="$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 1024 )); then
    echo "[RECOVERY] GPU $gpu still uses ${used} MiB; requesting GPU reset"
    if ! sudo nvidia-smi --gpu-reset -i "$gpu"; then
      echo "[ERROR] GPU $gpu reset failed; run sudo reboot, then rerun this script" >&2
      exit 2
    fi
  fi
done

echo "[RECOVERY] GPUs are available; resuming the existing experiment artifacts"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
exec env GPU_IDS="$GPU_IDS" bash local_run/run_tofu_1p5b_4090x2.sh "$@"
