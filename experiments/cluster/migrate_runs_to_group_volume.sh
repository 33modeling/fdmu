#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT/runs"
TARGET="/group-volume/fdmu/runs"

log() {
  printf '[migrate-runs] %s\n' "$*"
}

fail() {
  printf '[migrate-runs] ERROR: %s\n' "$*" >&2
  exit 2
}

if pgrep -af "experiments/cluster/(worker.py|node_watch.py|monitor_queue.py)" \
    >/dev/null 2>&1; then
  pgrep -af "experiments/cluster/(worker.py|node_watch.py|monitor_queue.py)" >&2
  fail "cluster processes are active; stop them before moving runs"
fi

mkdir -p "$TARGET" || fail "cannot create target: $TARGET"
[[ -w "$TARGET" ]] || fail "target is not writable: $TARGET"

if [[ -L "$SOURCE" ]]; then
  current="$(readlink -f "$SOURCE")"
  [[ "$current" == "$TARGET" ]] \
    || fail "$SOURCE already points to a different target: $current"
  log "already migrated: $SOURCE -> $TARGET"
  exit 0
fi

if [[ -d "$SOURCE" ]]; then
  log "source usage before migration:"
  du -sh "$SOURCE"
  log "moving files to $TARGET"
  PYTHON="/group-volume/fdmu/.venv/bin/python"
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)" \
      || fail "python3 is required when the group-volume venv is unavailable"
  fi
  "$PYTHON" experiments/cluster/move_runs.py "$SOURCE" "$TARGET" \
    || fail "Python migration failed; source files were not discarded"
fi

[[ ! -e "$SOURCE" ]] || fail "source path is not empty: $SOURCE"
ln -s "$TARGET" "$SOURCE"

log "migration complete: $SOURCE -> $TARGET"
du -sh "$TARGET"
df -h "$ROOT" "$TARGET"
