"""Stream one model's queue progress and surface failed unit logs."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERROR_TOKENS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "refusing",
    "permission denied",
    "no space left",
    "out of memory",
    "modulenotfounderror",
    "assertionerror",
)


def ids_in(queue: Path, state: str, needle: str) -> list[str]:
    directory = queue / state
    if not directory.exists():
        return []
    return sorted(
        path.stem
        for path in directory.glob("*.json")
        if not path.name.endswith(".meta.json") and needle in path.stem
    )


def tail(path: Path, lines: int) -> str:
    if not path.is_file():
        return "(log file missing)"
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines)).rstrip()


def error_summary(path: Path | None, scan_lines: int = 400) -> list[str]:
    if path is None or not path.is_file():
        return ["log file is missing"]
    lines = tail(path, scan_lines).splitlines()
    selected = [
        line.strip()
        for line in lines
        if any(token in line.lower() for token in ERROR_TOKENS)
    ]
    unique: list[str] = []
    for line in selected:
        if line and line not in unique:
            unique.append(line)
    return unique[-8:] or ["no recognized error marker; inspect the retained log tail"]


def failed_log(queue: Path, unit_id: str) -> Path | None:
    payload_path = queue / "failed" / f"{unit_id}.json"
    if not payload_path.is_file():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    raw = (payload.get("result") or {}).get("log")
    return Path(raw) if raw else None


def latest_running_log(unit_id: str) -> Path | None:
    runs_root = Path(
        os.environ.get(
            "CLUSTER_RUNS_ROOT",
            "/group-volume/fdmu/runs",
        )
    )
    matches = list((runs_root / "logs/cluster").glob(f"{unit_id}__*.out"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def heartbeat_age(queue: Path, unit_id: str) -> float | None:
    heartbeat = queue / "claimed" / f"{unit_id}.hb"
    claim = queue / "claimed" / f"{unit_id}.json"
    for stamp in (heartbeat, claim):
        try:
            return max(0.0, time.time() - stamp.stat().st_mtime)
        except FileNotFoundError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--tail-lines", type=int, default=30)
    parser.add_argument(
        "--stale-after",
        type=float,
        default=1800.0,
        help="fail when a claimed unit heartbeat exceeds this age",
    )
    args = parser.parse_args()
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")

    while True:
        states = {
            state: ids_in(args.queue, state, args.match)
            for state in ("pending", "claimed", "done", "failed")
        }
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(
            f"[INFO] [{now}] queue={args.queue} model={args.match} "
            f"pending={len(states['pending'])} running={len(states['claimed'])} "
            f"done={len(states['done'])} failed={len(states['failed'])}",
            flush=True,
        )

        if states["failed"]:
            for unit_id in states["failed"]:
                log = failed_log(args.queue, unit_id)
                print(
                    f"[ANALYSIS] unit={unit_id} likely_failure_lines:",
                    flush=True,
                )
                for line in error_summary(log):
                    print(f"[ANALYSIS]   {line}", flush=True)
                print(
                    f"\n[ERROR] ===== BEGIN FAILED UNIT LOG "
                    f"unit={unit_id} path={log} =====",
                    flush=True,
                )
                print(tail(log, args.tail_lines) if log else "(no log path)", flush=True)
                print(
                    f"[ERROR] ===== END FAILED UNIT LOG unit={unit_id} =====",
                    flush=True,
                )
            return 1

        for unit_id in states["claimed"]:
            age = heartbeat_age(args.queue, unit_id)
            if age is None:
                print(
                    f"[INFO] unit={unit_id} changed state during polling",
                    flush=True,
                )
                continue
            if age > args.stale_after:
                print(
                    f"[ERROR] STALE CLAIM unit={unit_id} heartbeat_age_s={age:.1f} "
                    f"threshold_s={args.stale_after:.1f}",
                    flush=True,
                )
                print(
                    "[ERROR] verify the owning worker is dead on its host, then run "
                    "workqueue.py requeue-stale; automatic requeue is intentionally disabled",
                    flush=True,
                )
                return 2
            log = latest_running_log(unit_id)
            last = tail(log, 1) if log else "(waiting for unit log)"
            print(
                f"[INFO] RUN unit={unit_id} heartbeat_age_s={age:.1f} "
                f"last_log={last}",
                flush=True,
            )

        if not states["pending"] and not states["claimed"]:
            print(f"[INFO] [{now}] model queue complete", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
