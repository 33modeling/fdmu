#!/usr/bin/env bash
# Shared state plus per-host scratch paths for H100 cluster jobs.

CLUSTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_LOCAL_ENV="${CLUSTER_LOCAL_ENV:-$CLUSTER_REPO_ROOT/.cluster_env.local.sh}"
export GROUP_VOLUME_ROOT="${FDMU_TEST_GROUP_VOLUME_ROOT:-/group-volume}"

# Queue/results and scratch live on group-volume, independently of where the
# checkout itself is mounted. A local config may customize storage paths only
# to another location below /group-volume.
unset \
  CLUSTER_STORAGE_ROOT \
  CLUSTER_RUNS_ROOT \
  CLUSTER_RUNTIME_BASE \
  CLUSTER_WORK_ROOT \
  CLUSTER_HF_HOME \
  CLUSTER_HF_DATASETS_CACHE \
  CLUSTER_RSUS_DATASETS_CACHE \
  CLUSTER_TORCH_HOME \
  CLUSTER_XDG_CACHE_HOME \
  CLUSTER_TMPDIR
if [[ -f "$CLUSTER_LOCAL_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$CLUSTER_LOCAL_ENV"
  printf '[cluster-env] loaded local overrides: %s\n' "$CLUSTER_LOCAL_ENV"
fi

CLUSTER_USER="${USER:-$(id -un)}"
CLUSTER_HOST="${HOSTNAME:-$(hostname)}"
export CLUSTER_STORAGE_ROOT="${CLUSTER_STORAGE_ROOT:-$GROUP_VOLUME_ROOT/jieuns.shin/fdmu}"
export CLUSTER_RUNS_ROOT="$CLUSTER_STORAGE_ROOT/runs"
CLUSTER_RUNTIME_BASE="${CLUSTER_RUNTIME_BASE:-$CLUSTER_STORAGE_ROOT/runtime}"
export CLUSTER_WORK_ROOT="$CLUSTER_RUNTIME_BASE/$CLUSTER_USER/$CLUSTER_HOST"

ensure_group_volume_path() {
  local role="$1"
  local path="$2"
  case "$path/" in
    "$GROUP_VOLUME_ROOT"/*) ;;
    *)
      printf 'cluster %s must be under %s, found: %s\n' \
        "$role" "$GROUP_VOLUME_ROOT" "$path" >&2
      return 1
      ;;
  esac
}

export HF_HOME="${CLUSTER_HF_HOME:-$GROUP_VOLUME_ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${CLUSTER_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RSUS_DATASETS_CACHE="${CLUSTER_RSUS_DATASETS_CACHE:-$CLUSTER_WORK_ROOT/datasets_cache}"
export TORCH_HOME="${CLUSTER_TORCH_HOME:-$CLUSTER_WORK_ROOT/torch_home}"
export XDG_CACHE_HOME="${CLUSTER_XDG_CACHE_HOME:-$CLUSTER_WORK_ROOT/xdg_cache}"
export TMPDIR="${CLUSTER_TMPDIR:-$CLUSTER_WORK_ROOT/tmp}"
export HOME="$CLUSTER_WORK_ROOT/home"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CONFIG_HOME="$CLUSTER_WORK_ROOT/xdg_config"
export XDG_DATA_HOME="$CLUSTER_WORK_ROOT/xdg_data"
export HF_HUB_CACHE="$CLUSTER_WORK_ROOT/huggingface/hub"
export HF_ASSETS_CACHE="$CLUSTER_WORK_ROOT/huggingface/assets"
export HF_MODULES_CACHE="$CLUSTER_WORK_ROOT/huggingface/modules"
export TRANSFORMERS_CACHE="$CLUSTER_WORK_ROOT/huggingface/transformers"
export TORCH_EXTENSIONS_DIR="$CLUSTER_WORK_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CLUSTER_WORK_ROOT/triton"
export CUDA_CACHE_PATH="$CLUSTER_WORK_ROOT/cuda"
export MPLCONFIGDIR="$CLUSTER_WORK_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$CLUSTER_WORK_ROOT/numba"
export PIP_CACHE_DIR="$CLUSTER_WORK_ROOT/pip"
export PYTHONPYCACHEPREFIX="$CLUSTER_WORK_ROOT/pycache"
export WANDB_DIR="$CLUSTER_WORK_ROOT/wandb"
export WANDB_CACHE_DIR="$CLUSTER_WORK_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$CLUSTER_WORK_ROOT/wandb/config"

if [[ ! -d "$HF_HOME" ]]; then
  printf 'shared HF_HOME is missing or not mounted: %s\n' "$HF_HOME" >&2
  return 2 2>/dev/null || exit 2
fi

for protected_path in \
  "$CLUSTER_RUNS_ROOT" \
  "$CLUSTER_WORK_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$RSUS_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR"; do
  ensure_group_volume_path "storage path" "$protected_path" \
    || { return 2 2>/dev/null || exit 2; }
done

ensure_writable_dir() {
  local role="$1"
  local path="$2"
  local probe
  if ! mkdir -p "$path"; then
    printf 'cluster %s directory cannot be created: %s\n' "$role" "$path" >&2
    return 1
  fi
  probe="$path/.fdmu-write-probe-${CLUSTER_HOST}-$$"
  if ! (umask 077 && printf '%s\n' "$$" > "$probe"); then
    printf 'cluster %s directory is not writable: %s\n' "$role" "$path" >&2
    return 1
  fi
  rm -f "$probe"
}

ensure_writable_dir "shared state" "$CLUSTER_RUNS_ROOT" \
  || { return 2 2>/dev/null || exit 2; }
for writable_path in \
  "$CLUSTER_WORK_ROOT" \
  "$RSUS_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$HF_MODULES_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR"; do
  ensure_writable_dir "per-host scratch" "$writable_path" \
    || { return 2 2>/dev/null || exit 2; }
done

printf '[cluster-env] state=%s scratch=%s home=%s tmp=%s hf_readonly=%s\n' \
  "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" "$HOME" "$TMPDIR" "$HF_HOME"
