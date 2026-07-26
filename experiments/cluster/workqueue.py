"""Shared-filesystem work queue for multi-node H100 campaigns.

Every node mounts the same repository under /group-volume, so a directory of
JSON files is the coordination substrate: no scheduler, no daemon, no extra
dependency.  A unit moves through ``pending -> claimed -> done|failed`` by
atomic rename.  Workers on any node claim units concurrently; a claim that
loses the rename race simply moves on to the next pending unit.

Layout under ``--queue <root>``::

    pending/<unit_id>.json      enqueued unit + attempt counter
    claimed/<unit_id>.json      unit currently owned by a worker
    claimed/<unit_id>.meta.json owner token (host/gpu/pid/started)
    claimed/<unit_id>.hb        heartbeat file (mtime refreshed by the owner)
    done/<unit_id>.json         unit + result of the successful attempt
    failed/<unit_id>.json       unit + result after max_attempts exhausted

Crash recovery: a worker that dies stops refreshing its heartbeat.
``workqueue.py requeue-stale`` moves such units back to ``pending``.  Requeue can
in principle double-run a unit that is still alive but silent, so every
enqueued command must be resume-safe (all campaign runners here take
``--resume``).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable

STATES = ("pending", "claimed", "done", "failed")


class ClaimLostError(RuntimeError):
    """Raised when a stale worker no longer owns its queue claim."""


@dataclasses.dataclass
class Unit:
    unit_id: str
    cmd: list[str]
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    gpus: int = 1
    max_attempts: int = 2
    code_commit: str = ""

    def to_payload(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_payload(payload: dict) -> "Unit":
        return Unit(
            unit_id=str(payload["unit_id"]),
            cmd=[str(part) for part in payload["cmd"]],
            env={str(k): str(v) for k, v in payload.get("env", {}).items()},
            gpus=int(payload.get("gpus", 1)),
            max_attempts=int(payload.get("max_attempts", 2)),
            code_commit=str(payload.get("code_commit", "")),
        )


@dataclasses.dataclass
class Claim:
    unit: Unit
    attempts: int  # attempts already consumed before this one
    token: str
    path: Path  # claimed/<unit_id>.json


def _validate_unit_id(unit_id: str) -> None:
    ok = unit_id and all(ch.isalnum() or ch in "._-" for ch in unit_id)
    if not ok:
        raise ValueError(f"unit_id must be filesystem-safe [A-Za-z0-9._-]: {unit_id!r}")


class WorkQueue:
    def __init__(self, root: Path):
        self.root = Path(root)

    def init(self) -> None:
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        (self.root / ".locks").mkdir(parents=True, exist_ok=True)

    def _state_dir(self, state: str) -> Path:
        return self.root / state

    def _entry(self, state: str, unit_id: str) -> Path:
        return self._state_dir(state) / f"{unit_id}.json"

    def _write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _locate(self, unit_id: str) -> str | None:
        for state in STATES:
            if self._entry(state, unit_id).exists():
                return state
        return None

    @contextmanager
    def _unit_lock(self, unit_id: str):
        self.init()
        path = self.root / ".locks" / f"{unit_id}.lock"
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _assert_claim_owner(self, claim: Claim) -> None:
        meta_path = claim.path.with_name(f"{claim.unit.unit_id}.meta.json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ClaimLostError(
                f"claim no longer exists for {claim.unit.unit_id}"
            ) from exc
        if not claim.path.exists() or metadata.get("token") != claim.token:
            raise ClaimLostError(
                f"claim token changed for {claim.unit.unit_id}; "
                "stale worker result was rejected"
            )

    # -- producer side ------------------------------------------------------

    def enqueue(
        self, units: Iterable[Unit], *, skip_existing: bool = False
    ) -> list[str]:
        """Enqueue units; queues stay append-only per unit_id.

        The default raises on the first duplicate — but note the earlier
        units of the same call are already on disk at that point.  Top-up
        callers (re-running an idempotent wave enqueue) should pass
        ``skip_existing=True`` so units after a duplicate are still added;
        skipped ids are reported by the CLI rather than silently dropped.
        """
        self.init()
        added = []
        skipped: list[str] = []
        for unit in units:
            _validate_unit_id(unit.unit_id)
            with self._unit_lock(unit.unit_id):
                state = self._locate(unit.unit_id)
                if state is not None:
                    if skip_existing:
                        skipped.append(unit.unit_id)
                        continue
                    raise FileExistsError(
                        f"unit {unit.unit_id} already exists in state {state!r}; "
                        "queues are append-only per unit_id — pick a new id"
                    )
                self._write_json(
                    self._entry("pending", unit.unit_id),
                    {"unit": unit.to_payload(), "attempts": 0},
                )
            added.append(unit.unit_id)
        self.last_skipped = skipped
        return added

    # -- worker side --------------------------------------------------------

    def claim(
        self,
        owner: dict | None = None,
        *,
        unit_id_contains: str = "",
        preferred_prefix: str = "",
    ) -> Claim | None:
        """Claim one pending unit, or return None when nothing is claimable."""
        pending = sorted(
            self._state_dir("pending").glob("*.json"),
            key=lambda path: (
                bool(preferred_prefix)
                and not path.stem.startswith(preferred_prefix),
                path.name,
            ),
        )
        for path in pending:
            if unit_id_contains and unit_id_contains not in path.stem:
                continue
            with self._unit_lock(path.stem):
                if not path.exists():
                    continue
                dst = self._entry("claimed", path.stem)
                if dst.exists():
                    raise RuntimeError(
                        f"queue invariant violated for {path.stem}: "
                        "unit exists in pending and claimed"
                    )
                os.replace(path, dst)
                token = uuid.uuid4().hex
                meta = dict(owner or {})
                meta.update({"token": token, "claimed_at": time.time()})
                meta_path = dst.with_name(f"{path.stem}.meta.json")
                self._write_json(meta_path, meta)
                observed = json.loads(meta_path.read_text(encoding="utf-8"))
                if observed.get("token") != token:
                    continue
                payload = json.loads(dst.read_text(encoding="utf-8"))
                claim = Claim(
                    unit=Unit.from_payload(payload["unit"]),
                    attempts=int(payload.get("attempts", 0)),
                    token=token,
                    path=dst,
                )
                self._assert_claim_owner(claim)
                hb = self._state_dir("claimed") / f"{path.stem}.hb"
                hb.touch()
                return claim
        return None

    def heartbeat(self, claim: Claim) -> None:
        with self._unit_lock(claim.unit.unit_id):
            self._assert_claim_owner(claim)
            hb = self._state_dir("claimed") / f"{claim.unit.unit_id}.hb"
            hb.touch()

    def _clear_claim(self, unit_id: str) -> None:
        for suffix in (".json", ".meta.json", ".hb"):
            path = self._state_dir("claimed") / f"{unit_id}{suffix}"
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def complete(self, claim: Claim, result: dict) -> None:
        self._finish(claim, "done", result)

    def fail(self, claim: Claim, result: dict) -> str:
        """Record a failed attempt.  Returns the resulting state."""
        with self._unit_lock(claim.unit.unit_id):
            self._assert_claim_owner(claim)
            attempts = claim.attempts + 1
            if attempts < claim.unit.max_attempts:
                self._write_json(
                    self._entry("pending", claim.unit.unit_id),
                    {"unit": claim.unit.to_payload(), "attempts": attempts,
                     "last_failure": result},
                )
                self._clear_claim(claim.unit.unit_id)
                return "pending"
            self._finish_locked(claim, "failed", result, attempts=attempts)
            return "failed"

    def _finish(self, claim: Claim, state: str, result: dict, attempts: int | None = None) -> None:
        with self._unit_lock(claim.unit.unit_id):
            self._assert_claim_owner(claim)
            self._finish_locked(claim, state, result, attempts)

    def _finish_locked(
        self, claim: Claim, state: str, result: dict, attempts: int | None = None
    ) -> None:
        self._write_json(
            self._entry(state, claim.unit.unit_id),
            {
                "unit": claim.unit.to_payload(),
                "attempts": claim.attempts + 1 if attempts is None else attempts,
                "result": result,
            },
        )
        self._clear_claim(claim.unit.unit_id)

    # -- maintenance --------------------------------------------------------

    def requeue_stale(self, max_age_s: float, now: float | None = None) -> list[str]:
        """Return claimed units whose heartbeat is older than max_age_s to pending."""
        now = time.time() if now is None else now
        requeued = []
        for path in sorted(self._state_dir("claimed").glob("*.json")):
            if path.name.endswith(".meta.json"):
                continue
            unit_id = path.stem
            with self._unit_lock(unit_id):
                if not path.exists():
                    continue
                hb = self._state_dir("claimed") / f"{unit_id}.hb"
                stamp = hb if hb.exists() else path
                if now - stamp.stat().st_mtime <= max_age_s:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._write_json(
                    self._entry("pending", unit_id),
                    {"unit": payload["unit"],
                     "attempts": int(payload.get("attempts", 0)) + 1,
                     "last_failure": {"reason": f"stale heartbeat > {max_age_s}s"}},
                )
                self._clear_claim(unit_id)
                requeued.append(unit_id)
        return requeued

    def cancel(self, unit_id: str) -> str:
        """Move a pending or claimed unit to failed with a cancelled marker.

        Cancelling a *claimed* unit does not stop its worker process — kill
        the process first, then cancel so the failure record is deliberate
        rather than a retry loop.
        """
        for state in ("pending", "claimed"):
            with self._unit_lock(unit_id):
                path = self._entry(state, unit_id)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._write_json(
                    self._entry("failed", unit_id),
                    {"unit": payload["unit"],
                     "attempts": int(payload.get("attempts", 0)),
                     "result": {"cancelled": True, "from_state": state}},
                )
                if state == "pending":
                    path.unlink()
                else:
                    self._clear_claim(unit_id)
                return state
        raise FileNotFoundError(f"unit {unit_id} is not pending or claimed")

    def retry_failed(
        self,
        unit_ids: set[str] | None = None,
        *,
        code_commit: str | None = None,
    ) -> list[str]:
        """Move selected failed units back to pending under their unit locks.

        ``code_commit`` is an explicit operator re-pin for retries after a code
        fix. With explicit unit IDs it also re-pins automatic-retry units that
        are already pending. Omitting it preserves the original execution pin.
        """
        if code_commit is not None:
            normalized = code_commit.strip().lower()
            if len(normalized) != 40 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ValueError(
                    "retry code_commit must be an explicit 40-character Git SHA"
                )
            code_commit = normalized
        retry_ids = (
            sorted(unit_ids)
            if unit_ids is not None
            else sorted(path.stem for path in self._state_dir("failed").glob("*.json"))
        )
        retried = []
        for unit_id in retry_ids:
            with self._unit_lock(unit_id):
                failed_path = self._entry("failed", unit_id)
                pending_path = self._entry("pending", unit_id)
                if failed_path.exists():
                    source_state = "failed"
                    source_path = failed_path
                elif (
                    code_commit is not None
                    and unit_ids is not None
                    and pending_path.exists()
                ):
                    source_state = "pending"
                    source_path = pending_path
                else:
                    continue
                conflicting = [
                    state
                    for state in STATES
                    if state != source_state and self._entry(state, unit_id).exists()
                ]
                if conflicting:
                    raise RuntimeError(
                        f"queue invariant violated for {unit_id}: {source_state} and "
                        f"{conflicting} entries coexist"
                    )
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                unit_payload = dict(payload["unit"])
                previous_commit = str(unit_payload.get("code_commit", ""))
                if code_commit is not None:
                    unit_payload["code_commit"] = code_commit
                pending_payload = {
                    "unit": unit_payload,
                    "attempts": 0,
                    "last_failure": (
                        payload.get("result", {})
                        if source_state == "failed"
                        else payload.get("last_failure", {})
                    ),
                }
                if code_commit is not None:
                    pending_payload["retry_pin"] = {
                        "source_state": source_state,
                        "previous_code_commit": previous_commit,
                        "code_commit": code_commit,
                    }
                self._write_json(
                    pending_path,
                    pending_payload,
                )
                if source_state == "failed":
                    failed_path.unlink()
            retried.append(unit_id)
        return retried

    def status(self) -> dict:
        report: dict = {"root": str(self.root), "counts": {}, "claimed": [], "failed": []}
        for state in STATES:
            entries = [
                p for p in self._state_dir(state).glob("*.json")
                if not p.name.endswith(".meta.json")
            ]
            report["counts"][state] = len(entries)
            if state == "claimed":
                now = time.time()
                for path in sorted(entries):
                    meta_path = path.with_name(f"{path.stem}.meta.json")
                    meta = (
                        json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta_path.exists() else {}
                    )
                    hb = self._state_dir("claimed") / f"{path.stem}.hb"
                    age = now - (hb.stat().st_mtime if hb.exists() else path.stat().st_mtime)
                    report["claimed"].append({
                        "unit_id": path.stem,
                        "host": meta.get("host"),
                        "gpu": meta.get("gpu"),
                        "heartbeat_age_s": round(age, 1),
                    })
            if state == "failed":
                for path in sorted(entries):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    report["failed"].append({
                        "unit_id": path.stem,
                        "exit_code": payload.get("result", {}).get("exit_code"),
                        "log": payload.get("result", {}).get("log"),
                    })
        return report


def read_units_jsonl(path: Path) -> list[Unit]:
    units = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            units.append(Unit.from_payload(json.loads(line)))
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "enqueue", "status", "requeue-stale",
                                           "retry-failed", "cancel"])
    parser.add_argument(
        "--unit",
        action="append",
        default=[],
        help="cancel/retry-failed: unit id(s) to target (repeatable)",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="enqueue: skip units already present in any state "
                             "instead of aborting mid-batch")
    parser.add_argument("--queue", required=True, help="queue root directory (on the shared volume)")
    parser.add_argument("--units", help="units JSONL file (enqueue)")
    parser.add_argument("--stale-after", type=float, default=1800.0,
                        help="requeue-stale: heartbeat age threshold in seconds")
    parser.add_argument(
        "--code-commit",
        help="retry-failed: explicitly re-pin retried units to this full Git SHA",
    )
    parser.add_argument("--brief", action="store_true",
                        help="status: one line per queue state instead of full JSON")
    args = parser.parse_args()

    queue = WorkQueue(Path(args.queue))
    if args.action == "init":
        queue.init()
        print(f"initialized {queue.root}")
    elif args.action == "enqueue" and args.skip_existing:
        if not args.units:
            parser.error("enqueue requires --units <file.jsonl>")
        added = queue.enqueue(
            read_units_jsonl(Path(args.units)), skip_existing=True
        )
        skipped = getattr(queue, "last_skipped", [])
        print(f"enqueued {len(added)} unit(s); skipped {len(skipped)} existing")
        for unit_id in skipped:
            print(f"  skipped: {unit_id}")
    elif args.action == "enqueue":
        if not args.units:
            parser.error("enqueue requires --units <file.jsonl>")
        added = queue.enqueue(read_units_jsonl(Path(args.units)))
        print(f"enqueued {len(added)} unit(s)")
        for unit_id in added:
            print(f"  {unit_id}")
    elif args.action == "status":
        report = queue.status()
        if not args.brief:
            print(json.dumps(report, indent=2))
        counts = report["counts"]
        total = sum(counts.values())
        done = counts.get("done", 0)
        print(f"progress: {done}/{total} done, {counts.get('claimed', 0)} running, "
              f"{counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed")
        if args.brief:
            for row in report["claimed"]:
                print(f"  RUN  {row['unit_id']}  {row['host']} gpu{row['gpu']} "
                      f"hb={row['heartbeat_age_s']}s")
            for row in report["failed"]:
                print(f"  FAIL {row['unit_id']}  exit={row['exit_code']}  {row['log']}")
    elif args.action == "requeue-stale":
        requeued = queue.requeue_stale(args.stale_after)
        print(f"requeued {len(requeued)} stale unit(s): {requeued}")
    elif args.action == "retry-failed":
        retried = queue.retry_failed(
            set(args.unit) if args.unit else None,
            code_commit=args.code_commit,
        )
        print(f"retried {len(retried)} failed unit(s): {retried}")
    elif args.action == "cancel":
        if not args.unit:
            parser.error("cancel requires at least one --unit <id>")
        for unit_id in args.unit:
            state = queue.cancel(unit_id)
            print(f"cancelled {unit_id} (was {state})")


if __name__ == "__main__":
    main()
