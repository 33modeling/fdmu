"""Re-pin one model's pending/failed queue units to a clean checkout commit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "cluster"))

from workqueue import WorkQueue  # noqa: E402


def _payload(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("unit"), dict):
        raise ValueError(f"invalid queue entry: {path}")
    return value


def reconcile(queue_root: Path, model_id: str, code_commit: str) -> list[str]:
    queue = WorkQueue(queue_root)
    claimed_mismatches = []
    for path in sorted((queue.root / "claimed").glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        payload = _payload(path)
        unit = payload["unit"]
        unit_id = str(unit.get("unit_id", path.stem))
        if model_id in unit_id and unit.get("code_commit") != code_commit:
            claimed_mismatches.append(
                f"{unit_id}:{unit.get('code_commit', 'legacy-unpinned')}"
            )
    if claimed_mismatches:
        raise RuntimeError(
            "cannot re-pin active model units; stop their owning workers first: "
            + ", ".join(claimed_mismatches)
        )

    stale_ids = set()
    for state in ("pending", "failed"):
        for path in sorted((queue.root / state).glob("*.json")):
            payload = _payload(path)
            unit = payload["unit"]
            unit_id = str(unit.get("unit_id", path.stem))
            if model_id in unit_id and unit.get("code_commit") != code_commit:
                stale_ids.add(unit_id)

    reconciled = queue.retry_failed(stale_ids, code_commit=code_commit)
    missing = sorted(stale_ids - set(reconciled))
    if missing:
        raise RuntimeError(
            f"queue units changed state during commit reconciliation: {missing}"
        )
    print(
        f"commit reconciliation model={model_id} commit={code_commit} "
        f"repinned={len(reconciled)} units={reconciled}",
        flush=True,
    )
    return reconciled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    commit = args.code_commit.strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        parser.error("--code-commit must be a full 40-character Git SHA")
    reconcile(args.queue, args.model_id, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
