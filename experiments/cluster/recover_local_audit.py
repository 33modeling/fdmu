"""Stop and recover local audit claims before rebuilding fidelity artifacts.

This is intentionally model- and queue-scoped.  It never touches a claim owned
by another host and it only releases a local claim after both its worker and
matching audit process group have stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workqueue import WorkQueue  # noqa: E402


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    pgid: int
    argv: tuple[str, ...]


def process_snapshot(proc_root: Path = Path("/proc")) -> dict[int, Process]:
    processes: dict[int, Process] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            argv = tuple(
                part.decode("utf-8", errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            )
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_comm = stat[stat.rfind(")") + 2 :].split()
            ppid = int(after_comm[1])
            pgid = int(after_comm[2])
        except (FileNotFoundError, OSError, ValueError):
            continue
        if argv:
            processes[pid] = Process(pid, ppid, pgid, argv)
    return processes


def _argument(argv: tuple[str, ...], name: str) -> str | None:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def is_queue_worker(process: Process, queue: Path) -> bool:
    if not any(
        part.endswith("experiments/cluster/worker.py") for part in process.argv
    ):
        return False
    queue_arg = _argument(process.argv, "--queue")
    return queue_arg is not None and Path(queue_arg).resolve() == queue.resolve()


def is_matching_audit(
    process: Process,
    *,
    config: Path,
    model_id: str,
) -> bool:
    if not any(
        part.endswith("experiments/channel_matrix/run_campaign.py")
        for part in process.argv
    ):
        return False
    config_arg = _argument(process.argv, "--config")
    if config_arg is None:
        return False
    candidate = Path(config_arg)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return (
        candidate.resolve() == config.resolve()
        and _argument(process.argv, "--phase") == "audit"
        and _argument(process.argv, "--model-id") == model_id
    )


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
            return False
    except (FileNotFoundError, OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_groups(
    processes: dict[int, Process],
    audit_pids: set[int],
    sig: signal.Signals,
) -> None:
    own_group = os.getpgrp()
    groups = {
        processes[pid].pgid
        for pid in audit_pids
        if pid in processes and processes[pid].pgid != own_group
    }
    for pgid in sorted(groups):
        try:
            os.killpg(pgid, sig)
            print(f"[RECOVERY] signal={sig.name} audit_process_group={pgid}", flush=True)
        except ProcessLookupError:
            pass


def _signal_workers(worker_pids: set[int], sig: signal.Signals) -> None:
    for pid in sorted(worker_pids):
        try:
            os.kill(pid, sig)
            print(f"[RECOVERY] signal={sig.name} worker_pid={pid}", flush=True)
        except ProcessLookupError:
            pass


def _claimed_audits(
    queue: Path,
    unit_prefix: str,
) -> list[tuple[str, dict]]:
    claimed = queue / "claimed"
    rows = []
    for claim_path in sorted(claimed.glob(f"{unit_prefix}*.json")):
        if claim_path.name.endswith(".meta.json"):
            continue
        meta_path = claim_path.with_name(f"{claim_path.stem}.meta.json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot verify owner for claimed audit {claim_path.stem}: {exc}"
            ) from exc
        rows.append((claim_path.stem, metadata))
    return rows


def recover(
    *,
    queue: Path,
    config: Path,
    model_id: str,
    unit_prefix: str,
    grace_seconds: float,
) -> list[str]:
    host = socket.gethostname()
    claims = _claimed_audits(queue, unit_prefix)
    remote = [
        (unit_id, metadata)
        for unit_id, metadata in claims
        if metadata.get("host") != host
    ]
    if remote:
        details = ", ".join(
            f"{unit_id}@{metadata.get('host')} pid={metadata.get('pid')}"
            for unit_id, metadata in remote
        )
        raise RuntimeError(
            "refusing local recovery while matching audit claims are owned by "
            f"another host: {details}"
        )

    local_claim_ids = [unit_id for unit_id, _ in claims]
    owner_pids = {
        int(metadata["pid"])
        for _, metadata in claims
        if str(metadata.get("pid", "")).isdigit()
    }
    snapshot = process_snapshot()
    worker_pids = {
        pid
        for pid in owner_pids
        if pid in snapshot and is_queue_worker(snapshot[pid], queue)
    }
    audit_pids = {
        pid
        for pid, process in snapshot.items()
        if is_matching_audit(process, config=config, model_id=model_id)
    }

    print(
        f"[RECOVERY] local_claims={local_claim_ids or 'none'} "
        f"workers={sorted(worker_pids) or 'none'} "
        f"audit_processes={sorted(audit_pids) or 'none'}",
        flush=True,
    )
    _signal_process_groups(snapshot, audit_pids, signal.SIGTERM)
    _signal_workers(worker_pids, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    targets = worker_pids | audit_pids
    while any(_alive(pid) for pid in targets) and time.monotonic() < deadline:
        time.sleep(0.25)

    remaining = {pid for pid in targets if _alive(pid)}
    if remaining:
        current = process_snapshot()
        remaining_audits = remaining & audit_pids
        remaining_workers = remaining & worker_pids
        _signal_process_groups(current, remaining_audits, signal.SIGKILL)
        _signal_workers(remaining_workers, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while any(_alive(pid) for pid in remaining) and time.monotonic() < deadline:
            time.sleep(0.1)

    survivors = sorted(pid for pid in targets if _alive(pid))
    if survivors:
        raise RuntimeError(
            f"local audit processes did not stop after TERM/KILL: {survivors}"
        )

    queue_state = WorkQueue(queue)
    released = []
    for unit_id in local_claim_ids:
        try:
            previous = queue_state.cancel(unit_id)
        except FileNotFoundError:
            continue
        if previous not in {"claimed", "pending"}:
            raise RuntimeError(
                f"unexpected state while releasing {unit_id}: {previous}"
            )
        released.append(unit_id)
        print(
            f"[RECOVERY] released invalid-fidelity audit unit={unit_id} "
            f"from_state={previous}",
            flush=True,
        )
    return released


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--grace-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.grace_seconds <= 0:
        parser.error("--grace-seconds must be positive")
    recover(
        queue=Path(args.queue).resolve(),
        config=Path(args.config).resolve(),
        model_id=args.model_id,
        unit_prefix=args.unit_prefix,
        grace_seconds=args.grace_seconds,
    )


if __name__ == "__main__":
    main()
