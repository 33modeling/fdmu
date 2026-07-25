"""Move retryable partial audit runs aside without touching completed/active units."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHANNEL = ROOT / "experiments" / "channel_matrix"
sys.path.insert(0, str(CHANNEL))

from run_campaign import _complete, _has_artifacts, _runtime_path  # noqa: E402


def unit_state(queue: Path, unit_id: str) -> str | None:
    for state in ("claimed", "done", "failed", "pending"):
        if (queue / state / f"{unit_id}.json").is_file():
            return state
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = _runtime_path(cfg["output_root"])
    objectives = cfg["audit"]["objectives"] + cfg["audit"].get(
        "stress_objectives", []
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    forensics = Path(
        os.environ.get("CLUSTER_RUNS_ROOT", "/group-volume/fdmu/runs")
    ) / "forensics" / "audit-partials"

    moved = 0
    for author in cfg["audit"]["authors"]:
        unit_id = f"aud__{args.model_id}__a{author}"
        state = unit_state(args.queue, unit_id)
        if state not in {"failed", "pending"}:
            continue
        request_dir = output_root / "audit" / args.model_id / f"tofu-a{author}"
        for seed in cfg["audit"]["seeds"]:
            run_dir = request_dir / f"seed-{seed}"
            if not _has_artifacts(run_dir) or _complete(
                run_dir, objectives, audit=True
            ):
                continue
            destination = (
                forensics
                / f"{stamp}__{unit_id}__seed-{seed}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"forensics destination exists: {destination}")
            shutil.move(str(run_dir), str(destination))
            print(
                f"QUARANTINED state={state} unit={unit_id} "
                f"source={run_dir} destination={destination}"
            )
            moved += 1
    print(f"quarantined_partial_audits={moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
