#!/usr/bin/env python3
"""Run one campaign phase across explicit local GPU IDs with live logs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "experiments" / "channel_matrix" / "run_campaign.py"


@dataclass
class Active:
    request: int
    gpu: str
    process: subprocess.Popen
    thread: threading.Thread
    log_path: Path
    started: float


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _pump(active: Active, events: queue.Queue) -> None:
    assert active.process.stdout is not None
    prefix = f"[GPU {active.gpu} request={active.request}] "
    with active.log_path.open("a", encoding="utf-8") as log:
        for line in active.process.stdout:
            rendered = prefix + line
            print(rendered, end="", flush=True)
            log.write(rendered)
            log.flush()
    events.put(active.request)


def _terminate(active: list[Active], reason: str) -> None:
    for item in active:
        if item.process.poll() is None:
            print(
                f"[CLEANUP] reason={reason} request={item.request} "
                f"gpu={item.gpu} pid={item.process.pid}",
                flush=True,
            )
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(
        item.process.poll() is None for item in active
    ):
        time.sleep(0.2)
    for item in active:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        item.thread.join(timeout=2)


def run_phase(
    *,
    config_path: Path,
    phase: str,
    gpu_ids: list[str],
    dry_run: bool,
) -> None:
    config = _load(config_path)
    requests = [int(value) for value in config[phase]["authors"]]
    if not requests:
        raise ValueError(f"{phase} request roster is empty")
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    output_root = Path(config["output_root"])
    log_root = output_root / "launcher_logs" / phase
    log_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[PHASE] name={phase} requests={requests} gpus={gpu_ids} "
        f"parallelism={min(len(requests), len(gpu_ids))} dry_run={dry_run}",
        flush=True,
    )

    pending = list(requests)
    free = list(gpu_ids)
    active: list[Active] = []
    events: queue.Queue = queue.Queue()
    failures: list[dict] = []
    completed: list[int] = []
    last_heartbeat = 0.0
    try:
        while pending or active:
            while pending and free:
                request = pending.pop(0)
                gpu = free.pop(0)
                log_path = log_root / f"request-{request:03d}.log"
                command = [
                    sys.executable,
                    "-u",
                    str(CAMPAIGN),
                    "--config",
                    str(config_path),
                    "--phase",
                    phase,
                    "--only-authors",
                    str(request),
                    "--resume",
                ]
                if dry_run:
                    command.append("--dry-run")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                print(
                    f"[LAUNCH] phase={phase} request={request} gpu={gpu} "
                    f"log={log_path}",
                    flush=True,
                )
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                placeholder = Active(
                    request=request,
                    gpu=gpu,
                    process=process,
                    thread=threading.current_thread(),
                    log_path=log_path,
                    started=time.monotonic(),
                )
                thread = threading.Thread(
                    target=_pump,
                    args=(placeholder, events),
                    daemon=True,
                    name=f"{phase}-{request}",
                )
                placeholder.thread = thread
                active.append(placeholder)
                thread.start()

            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                running = [
                    {
                        "request": item.request,
                        "gpu": item.gpu,
                        "elapsed_seconds": round(now - item.started),
                    }
                    for item in active
                ]
                print(
                    f"[PROGRESS] phase={phase} completed={len(completed)}/"
                    f"{len(requests)} pending={len(pending)} "
                    f"running={json.dumps(running, sort_keys=True)}",
                    flush=True,
                )
                last_heartbeat = now

            try:
                finished_request = events.get(timeout=1.0)
            except queue.Empty:
                continue
            item = next(
                current
                for current in active
                if current.request == finished_request
            )
            item.thread.join(timeout=2)
            code = item.process.wait()
            active.remove(item)
            free.append(item.gpu)
            free.sort(key=gpu_ids.index)
            elapsed = round(time.monotonic() - item.started)
            if code:
                failures.append(
                    {
                        "request": item.request,
                        "gpu": item.gpu,
                        "exit_code": code,
                        "log": str(item.log_path),
                    }
                )
                print(
                    f"[ERROR] phase={phase} request={item.request} gpu={item.gpu} "
                    f"exit={code} elapsed_seconds={elapsed} log={item.log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate(active, f"{phase}-peer-failed")
                active.clear()
                break
            completed.append(item.request)
            print(
                f"[COMPLETE] phase={phase} request={item.request} gpu={item.gpu} "
                f"elapsed_seconds={elapsed}",
                flush=True,
            )
    except BaseException:
        _terminate(active, f"{phase}-interrupted")
        raise

    summary = {
        "phase": phase,
        "requests": requests,
        "completed": sorted(completed),
        "failures": failures,
        "gpu_ids": gpu_ids,
        "dry_run": dry_run,
    }
    summary_path = log_root / "LAST_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{phase} failed for request {failures[0]['request']}; "
            f"see {failures[0]['log']}"
        )
    print(
        f"[PHASE COMPLETE] name={phase} requests={len(completed)} "
        f"summary={summary_path}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "audit"), required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        gpu_ids = [
            value.strip()
            for value in args.gpu_ids.split(",")
            if value.strip()
        ]
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"duplicate GPU IDs: {gpu_ids}")
        run_phase(
            config_path=args.config.resolve(),
            phase=args.phase,
            gpu_ids=gpu_ids,
            dry_run=args.dry_run,
        )
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(
            f"parallel phase failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
