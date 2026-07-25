#!/usr/bin/env bash
# Shared state plus per-host scratch paths for H100 cluster jobs.

CLUSTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_LOCAL_ENV="${CLUSTER_LOCAL_ENV:-$CLUSTER_REPO_ROOT/.cluster_env.local.sh}"

# Queue/results state is part of the checkout's shared runs tree and must never
# drift through an inherited shell or queued unit environment. A local config
# may customize scratch/cache paths, but not campaign state.
unset \
  GROUP_VOLUME_ROOT \
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

GROUP_VOLUME_ROOT="${GROUP_VOLUME_ROOT:-/group-volume}"
CLUSTER_USER="${USER:-$(id -un)}"
CLUSTER_HOST="${HOSTNAME:-$(hostname)}"
export CLUSTER_RUNS_ROOT="$CLUSTER_REPO_ROOT/runs"
CLUSTER_RUNTIME_BASE="${CLUSTER_RUNTIME_BASE:-$CLUSTER_REPO_ROOT/.cluster-runtime}"
export CLUSTER_WORK_ROOT="$CLUSTER_RUNTIME_BASE/$CLUSTER_USER/$CLUSTER_HOST"

export HF_HOME="${CLUSTER_HF_HOME:-$GROUP_VOLUME_ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${CLUSTER_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RSUS_DATASETS_CACHE="${CLUSTER_RSUS_DATASETS_CACHE:-$CLUSTER_WORK_ROOT/datasets_cache}"
export TORCH_HOME="${CLUSTER_TORCH_HOME:-$CLUSTER_WORK_ROOT/torch_home}"
export XDG_CACHE_HOME="${CLUSTER_XDG_CACHE_HOME:-$CLUSTER_WORK_ROOT/xdg_cache}"
export TMPDIR="${CLUSTER_TMPDIR:-$CLUSTER_WORK_ROOT/tmp}"

if [[ ! -d "$HF_HOME" ]]; then
  printf 'shared HF_HOME is missing or not mounted: %s\n' "$HF_HOME" >&2
  return 2 2>/dev/null || exit 2
fi

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
  "$TMPDIR"; do
  ensure_writable_dir "per-host scratch" "$writable_path" \
    || { return 2 2>/dev/null || exit 2; }
done

printf '[cluster-env] state=%s scratch=%s hf_readonly=%s\n' \
  "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" "$HF_HOME"
