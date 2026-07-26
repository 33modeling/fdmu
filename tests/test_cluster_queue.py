"""CPU-only tests for the shared-filesystem cluster queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "cluster"))

import make_units  # noqa: E402
import monitor_queue  # noqa: E402
import quarantine_failed_audit  # noqa: E402
import recover_local_audit  # noqa: E402
import worker  # noqa: E402
from workqueue import ClaimLostError, Unit, WorkQueue  # noqa: E402


def _unit(unit_id: str, cmd: list[str] | None = None, **kw) -> Unit:
    return Unit(unit_id=unit_id, cmd=cmd or ["true"], **kw)


def test_enqueue_claim_complete_roundtrip(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    assert q.status()["counts"] == {"pending": 2, "claimed": 0, "done": 0, "failed": 0}

    first = q.claim(owner={"host": "n1", "gpu": 0})
    assert first is not None and first.unit.unit_id == "a"
    second = q.claim(owner={"host": "n1", "gpu": 1})
    assert second is not None and second.unit.unit_id == "b"
    assert q.claim() is None

    q.complete(first, {"exit_code": 0})
    q.complete(second, {"exit_code": 0})
    counts = q.status()["counts"]
    assert counts["done"] == 2 and counts["claimed"] == 0


def test_enqueue_refuses_duplicate_unit_id_in_any_state(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a")])
    with pytest.raises(FileExistsError):
        q.enqueue([_unit("a")])
    claim = q.claim()
    q.complete(claim, {"exit_code": 0})
    with pytest.raises(FileExistsError):
        q.enqueue([_unit("a")])


def test_unit_id_must_be_filesystem_safe(tmp_path):
    q = WorkQueue(tmp_path / "q")
    with pytest.raises(ValueError):
        q.enqueue([_unit("bad/id")])


def test_fail_requeues_until_max_attempts(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=2)])

    claim = q.claim()
    assert q.fail(claim, {"exit_code": 1}) == "pending"
    counts = q.status()["counts"]
    assert counts["pending"] == 1 and counts["failed"] == 0

    claim = q.claim()
    assert claim.attempts == 1
    assert q.fail(claim, {"exit_code": 1}) == "failed"
    report = q.status()
    assert report["counts"]["failed"] == 1
    assert report["failed"][0]["exit_code"] == 1


def test_retry_failed_restores_full_attempt_budget(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=1)])
    q.fail(q.claim(), {"exit_code": 3})
    assert q.status()["counts"]["failed"] == 1
    assert q.retry_failed() == ["a"]
    claim = q.claim()
    assert claim is not None and claim.attempts == 0


def test_retry_failed_can_target_specific_units(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=1), _unit("b", max_attempts=1)])
    q.fail(q.claim(), {"exit_code": 2})
    q.fail(q.claim(), {"exit_code": 2})
    assert q.retry_failed({"b"}) == ["b"]
    report = q.status()
    assert report["counts"]["pending"] == 1
    assert [row["unit_id"] for row in report["failed"]] == ["a"]


def test_retry_failed_can_explicitly_repin_current_commit(tmp_path):
    q = WorkQueue(tmp_path / "q")
    old_commit = "1" * 40
    current_commit = "2" * 40
    q.enqueue([_unit("a", max_attempts=1, code_commit=old_commit)])
    q.fail(q.claim(), {"exit_code": 2})

    assert q.retry_failed({"a"}, code_commit=current_commit) == ["a"]
    payload = json.loads(
        (q.root / "pending" / "a.json").read_text(encoding="utf-8")
    )
    assert payload["unit"]["code_commit"] == current_commit
    assert payload["retry_pin"] == {
        "source_state": "failed",
        "previous_code_commit": old_commit,
        "code_commit": current_commit,
    }
    with pytest.raises(ValueError, match="40-character Git SHA"):
        q.retry_failed(code_commit="short")


def test_retry_repin_updates_targeted_pending_unit(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=2, code_commit="1" * 40)])
    assert q.fail(q.claim(), {"exit_code": 2}) == "pending"

    assert q.retry_failed({"a"}, code_commit="2" * 40) == ["a"]
    payload = json.loads(
        (q.root / "pending" / "a.json").read_text(encoding="utf-8")
    )
    assert payload["attempts"] == 0
    assert payload["unit"]["code_commit"] == "2" * 40
    assert payload["retry_pin"]["source_state"] == "pending"


def test_enqueue_serializes_same_unit_producers(tmp_path):
    first_write_entered = threading.Event()
    release_first_write = threading.Event()
    write_guard = threading.Lock()
    first_pending_write = {"seen": False}

    class SlowQueue(WorkQueue):
        def _write_json(self, path, payload):
            if path.parent.name == "pending":
                with write_guard:
                    should_wait = not first_pending_write["seen"]
                    first_pending_write["seen"] = True
                if should_wait:
                    first_write_entered.set()
                    assert release_first_write.wait(timeout=2)
            return super()._write_json(path, payload)

    q = SlowQueue(tmp_path / "q")
    results: list[list[str]] = []

    def enqueue() -> None:
        results.append(q.enqueue([_unit("same")], skip_existing=True))

    first = threading.Thread(target=enqueue)
    second = threading.Thread(target=enqueue)
    first.start()
    assert first_write_entered.wait(timeout=2)
    second.start()
    release_first_write.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(results, key=len) == [[], ["same"]]
    assert q.status()["counts"]["pending"] == 1


def test_claim_filter_leaves_unrelated_units_pending(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("other-model"), _unit("aud__qwen25_14b__a181")])

    claim = q.claim(unit_id_contains="qwen25_14b")
    assert claim is not None
    assert claim.unit.unit_id == "aud__qwen25_14b__a181"
    assert q.claim(unit_id_contains="qwen25_14b") is None
    assert (q.root / "pending" / "other-model.json").is_file()
    q.complete(claim, {"exit_code": 0})


def test_claim_prefers_audit_before_other_matching_model_units(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([
        _unit("alpha-dev__qwen25_14b__a198__s2025"),
        _unit("aud__qwen25_14b__a181"),
    ])

    first = q.claim(
        unit_id_contains="qwen25_14b",
        preferred_prefix="aud__qwen25_14b",
    )
    assert first is not None
    assert first.unit.unit_id == "aud__qwen25_14b__a181"
    q.complete(first, {"exit_code": 0})
    second = q.claim(
        unit_id_contains="qwen25_14b",
        preferred_prefix="aud__qwen25_14b",
    )
    assert second is not None
    assert second.unit.unit_id.startswith("alpha-dev__")


def test_requeue_stale_by_heartbeat_age(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    q.claim(owner={"host": "n1", "gpu": 0})
    q.claim(owner={"host": "n2", "gpu": 0})

    # Viewed from one hour in the future, only b has kept beating.
    future = time.time() + 3600
    os.utime(q.root / "claimed" / "b.hb", times=(future, future))
    requeued = q.requeue_stale(max_age_s=1800, now=future)
    assert requeued == ["a"]
    counts = q.status()["counts"]
    assert counts["pending"] == 1 and counts["claimed"] == 1
    payload = json.loads((q.root / "pending" / "a.json").read_text(encoding="utf-8"))
    assert payload["attempts"] == 0


def test_monitor_heartbeat_age_tolerates_claim_state_transition(tmp_path):
    queue = tmp_path / "q"
    claimed = queue / "claimed"
    claimed.mkdir(parents=True)
    assert monitor_queue.heartbeat_age(queue, "a") is None

    claim = claimed / "a.json"
    claim.write_text("{}", encoding="utf-8")
    old = time.time() - 120
    os.utime(claim, times=(old, old))
    assert monitor_queue.heartbeat_age(queue, "a") >= 119

    heartbeat = claimed / "a.hb"
    heartbeat.touch()
    assert monitor_queue.heartbeat_age(queue, "a") < 5


def test_stale_worker_cannot_finish_or_heartbeat_reclaimed_unit(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a")])
    old = q.claim(owner={"host": "old", "gpu": 0})
    assert old is not None

    future = time.time() + 3600
    assert q.requeue_stale(max_age_s=1800, now=future) == ["a"]
    current = q.claim(owner={"host": "new", "gpu": 1})
    assert current is not None and current.token != old.token

    with pytest.raises(ClaimLostError):
        q.heartbeat(old)
    with pytest.raises(ClaimLostError):
        q.complete(old, {"exit_code": 0})

    assert (q.root / "claimed" / "a.json").is_file()
    q.complete(current, {"exit_code": 0})
    assert (q.root / "done" / "a.json").is_file()


def test_claim_survives_concurrent_double_rename_semantics(tmp_path):
    # The losing side of a claim race sees FileNotFoundError on rename and
    # must move on to the next unit rather than crash.
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    stolen = q.claim()
    assert stolen.unit.unit_id == "a"
    nxt = q.claim()
    assert nxt is not None and nxt.unit.unit_id == "b"


def test_worker_env_isolates_one_gpu_per_unit():
    storage = "/group-volume/fdmu"
    base = {
        "PATH": "/bin",
        "CLUSTER_STORAGE_ROOT": storage,
        "CLUSTER_RUNS_ROOT": f"{storage}/runs",
        "CLUSTER_WORK_ROOT": f"{storage}/runtime/researcher/node-a",
    }
    env = worker.build_env(base, {"MODEL_ID": "qwen25_7b"}, gpu=5, needs_gpu=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["GPU"] == "5"
    assert env["MODEL_ID"] == "qwen25_7b"
    cpu_env = worker.build_env(base, {}, gpu=5, needs_gpu=False)
    assert "CUDA_VISIBLE_DEVICES" not in cpu_env
    assert cpu_env["TMPDIR"].startswith(f"{storage}/runtime/")
    assert cpu_env["RSUS_DATASETS_CACHE"].startswith(f"{storage}/runtime/")
    assert cpu_env["HOME"].startswith(f"{storage}/runtime/")


def test_worker_pins_queued_python_to_active_interpreter():
    assert worker.resolve_unit_command(["python", "-u", "job.py"]) == [
        sys.executable,
        "-u",
        "job.py",
    ]
    assert worker.resolve_unit_command(["python3", "job.py"])[0] == sys.executable
    assert worker.resolve_unit_command(["bash", "job.sh"]) == ["bash", "job.sh"]


def test_worker_env_replaces_node_local_tmpdir():
    env = worker.build_env(
        {
            "USER": "researcher",
            "HOSTNAME": "node-7b",
            "CLUSTER_RUNS_ROOT": "/group-volume/shared/runs",
            "CLUSTER_WORK_ROOT": "/group-volume/shared/scratch/researcher/node-7b",
            "TMPDIR": "/tmp",
            "HF_HOME": "/home/researcher/.cache/huggingface",
            "TORCH_HOME": "/home/researcher/.cache/torch",
        },
        {},
        gpu=0,
        needs_gpu=True,
    )
    runtime = "/group-volume/fdmu/runtime/researcher/node-7b"
    assert env["CLUSTER_RUNS_ROOT"] == "/group-volume/fdmu/runs"
    assert env["TMPDIR"] == f"{runtime}/tmp"
    assert env["HOME"] == f"{runtime}/home"
    assert env["HF_HOME"] == "/group-volume/data/hf_home"
    assert env["TORCH_HOME"] == f"{runtime}/torch_home"
    assert env["XDG_CACHE_HOME"] == f"{runtime}/xdg_cache"
    assert env["TRITON_CACHE_DIR"] == f"{runtime}/triton"
    assert env["CUDA_CACHE_PATH"] == f"{runtime}/cuda"


def test_worker_unit_env_cannot_redirect_shared_cluster_roots():
    env = worker.build_env(
        {
            "CLUSTER_RUNS_ROOT": "/group-volume/shared/runs",
            "CLUSTER_WORK_ROOT": "/group-volume/shared/scratch/node-a",
        },
        {
            "CLUSTER_RUNS_ROOT": "/group-volume/wrong/14b",
            "CLUSTER_WORK_ROOT": "/group-volume/wrong/runtime",
            "CLUSTER_TMPDIR": "/tmp/wrong",
        },
        gpu=0,
        needs_gpu=True,
    )
    runtime = f"/group-volume/fdmu/runtime/unknown/{worker.socket.gethostname()}"
    assert env["CLUSTER_RUNS_ROOT"] == "/group-volume/fdmu/runs"
    assert env["CLUSTER_WORK_ROOT"] == runtime
    assert env["TMPDIR"] == f"{runtime}/tmp"


def test_worker_ignores_user_volume_storage():
    env = worker.build_env(
        {
            "USER": "researcher",
            "HOSTNAME": "node-a",
            "CLUSTER_RUNS_ROOT": "/user-volume/researcher/runs",
            "CLUSTER_WORK_ROOT": "/user-volume/researcher/runtime",
        },
        {},
        gpu=0,
        needs_gpu=True,
    )
    assert env["CLUSTER_RUNS_ROOT"] == "/group-volume/fdmu/runs"
    assert env["CLUSTER_WORK_ROOT"] == "/group-volume/fdmu/runtime/researcher/node-a"


def test_model_launchers_pin_queues_without_force_override():
    launch = (ROOT / "experiments/cluster/launch_node.sh").read_text(encoding="utf-8")
    seven = (ROOT / "experiments/cluster/run_tofu_7b_h100.sh").read_text(
        encoding="utf-8"
    )
    fourteen = (ROOT / "experiments/cluster/run_tofu_14b_h100.sh").read_text(
        encoding="utf-8"
    )
    assert "FORCE_QUEUE" not in launch
    assert "FORCE_QUEUE" not in seven
    assert "FORCE_QUEUE" not in fourteen
    assert 'QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave2"' in seven
    assert 'QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave1_14b"' in fourteen
    assert 'launch_node.sh --dedicated-queue "$QUEUE"' in seven
    assert "WORKER_GPU=0" in fourteen
    assert 'launch_node.sh --dedicated-queue "$QUEUE" 1' in fourteen
    assert "experiments/cluster/monitor_queue.py" in fourteen
    assert "setup_group_volume.sh" in seven
    assert "setup_group_volume.sh" in fourteen
    assert 'bash "$ROOT/experiments/cluster/setup_group_volume.sh"' in seven
    assert 'bash "$ROOT/experiments/cluster/setup_group_volume.sh"' in fourteen
    assert "stage aggregate-latex" in seven
    assert "stage aggregate-latex" in fourteen
    assert "h100_campaign.sh aggregate" in seven
    assert "h100_campaign.sh aggregate" in fourteen
    assert "one dedicated worker on GPU 0" in fourteen
    assert seven.count("--unit aud__qwen25_7b__") == 3
    assert 'RETRY_ARGS+=(--unit "$unit_id")' in fourteen
    assert 'WAIT=0 UNIT_MATCH="$MODEL_ID" UNIT_PREFER="$AUDIT_MATCH"' in seven
    assert 'WAIT=0 UNIT_MATCH="$MODEL_ID" UNIT_PREFER="$AUDIT_MATCH"' in fourteen
    assert 'AUDIT_MATCH="aud__${MODEL_ID}"' in seven
    assert 'AUDIT_MATCH="aud__${MODEL_ID}"' in fourteen
    assert '--queue "$QUEUE" --match "$AUDIT_MATCH"' in seven
    assert '--queue "$QUEUE" --match "$AUDIT_MATCH"' in fourteen
    assert "alpha jobs continue independently" in seven
    assert "alpha jobs continue independently" in fourteen
    assert '--code-commit "$CURRENT_COMMIT"' in seven
    assert '--code-commit "$CURRENT_COMMIT"' in fourteen
    assert seven.index("stage retry-commit-validation") < seven.index(
        "stage failed-audit-recovery"
    )
    assert fourteen.index("stage retry-commit-validation") < fourteen.index(
        "stage failed-audit-recovery"
    )
    assert "refusing queue re-pin from a dirty worktree" in seven
    assert "refusing queue re-pin from a dirty worktree" in fourteen
    assert '--match-unit "${UNIT_MATCH}"' in launch
    assert '--prefer-unit-prefix "${UNIT_PREFER}"' in launch
    assert "HOST_LAUNCH_LOCK" in launch
    assert "HOST_LAUNCH_LOCK_TIMEOUT_SECONDS:-60" in launch
    assert launch.count("8>&- >>") == 2
    assert '8>&- >> "${out}" 2>&1 &' in launch
    assert "worker died during startup" in launch
    enqueue = (ROOT / "experiments/cluster/enqueue_table12.sh").read_text(
        encoding="utf-8"
    )
    assert "require_passed_fidelity" in enqueue
    assert 'require_passed_fidelity "${cfg}" qwen25_7b' in enqueue
    assert 'require_passed_fidelity "${cfg}" qwen25_14b' in enqueue
    assert "each 8-GPU node starts 8 workers" not in enqueue
    assert "exactly one dedicated worker on GPU 0" in enqueue
    setup = (ROOT / "experiments/cluster/setup_group_volume.sh").read_text(
        encoding="utf-8"
    )
    assert "sentence-transformers" not in setup
    assert 'flock -x -w "$LOCK_TIMEOUT_SECONDS" 9' in setup
    assert "setup.lock.owner" in setup
    assert "existing environment is ready; lock not required" in setup
    assert "environment was prepared by another host while waiting" in setup
    assert '"torch==2.7.1"' in setup
    assert '"transformers==4.53.2"' in setup
    assert '"datasets==2.19.2"' in setup
    assert '"PyYAML==6.0.2"' in setup


def test_quarantine_moves_only_retryable_partial_audit(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    queue = runs / "cluster_queue" / "wave2"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "aud__qwen25_7b__a181.json").write_text(
        "{}", encoding="utf-8"
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_root": "runs/channel_matrix_7b",
                "audit": {
                    "authors": [181],
                    "seeds": [2025],
                    "objectives": ["npo"],
                },
            }
        ),
        encoding="utf-8",
    )
    partial = (
        runs
        / "channel_matrix_7b"
        / "audit"
        / "qwen25_7b"
        / "tofu-a181"
        / "seed-2025"
    )
    partial.mkdir(parents=True)
    (partial / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CLUSTER_RUNS_ROOT", str(runs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quarantine_failed_audit.py",
            "--queue",
            str(queue),
            "--config",
            str(config),
            "--model-id",
            "qwen25_7b",
        ],
    )

    assert quarantine_failed_audit.main() == 0
    assert not partial.exists()
    moved = list((runs / "forensics" / "audit-partials").iterdir())
    assert len(moved) == 1
    assert (moved[0] / "run_manifest.json").is_file()


def test_worker_executes_units_and_records_results(tmp_path):
    q = WorkQueue(tmp_path / "q")
    marker = tmp_path / "ran.txt"
    q.enqueue([
        Unit(unit_id="ok", cmd=["sh", "-c", f"echo hello > {marker.as_posix()}"], gpus=0),
        Unit(unit_id="boom", cmd=["sh", "-c", "exit 7"], gpus=0, max_attempts=1),
    ])
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    while (claim := q.claim(owner={"host": "test", "gpu": -1})) is not None:
        worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)

    report = q.status()
    assert report["counts"]["done"] == 1 and report["counts"]["failed"] == 1
    assert marker.read_text().strip() == "hello"
    assert report["failed"][0]["exit_code"] == 7
    failed_log = Path(report["failed"][0]["log"])
    assert failed_log.exists() and "boom" in failed_log.name


def test_worker_survives_unlaunchable_command(tmp_path):
    # A typo'd custom unit (missing binary) must be recorded as a failure,
    # not crash the worker and orphan the claim.
    q = WorkQueue(tmp_path / "q")
    q.enqueue([
        Unit(unit_id="typo", cmd=["/no/such/binary"], gpus=0, max_attempts=1),
        Unit(unit_id="after", cmd=["true"], gpus=0),
    ])
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    while (claim := q.claim(owner={"host": "test", "gpu": -1})) is not None:
        worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)

    report = q.status()
    assert report["counts"] == {"pending": 0, "claimed": 0, "done": 1, "failed": 1}
    failed = json.loads((q.root / "failed" / "typo.json").read_text(encoding="utf-8"))
    assert failed["result"]["exit_code"] is None
    assert "FileNotFoundError" in failed["result"]["error"]


def test_worker_rejects_unit_pinned_to_another_commit(tmp_path):
    q = WorkQueue(tmp_path / "q")
    marker = tmp_path / "must-not-run"
    q.enqueue([
        Unit(
            unit_id="wrong-commit",
            cmd=["sh", "-c", f"touch {marker}"],
            gpus=0,
            max_attempts=1,
            code_commit="0" * 40,
        )
    ])
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    claim = q.claim(owner={"host": "test", "gpu": -1})
    assert claim is not None
    assert not worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)
    assert not marker.exists()
    failed = json.loads(
        (q.root / "failed" / "wrong-commit.json").read_text(encoding="utf-8")
    )
    assert "CodeCommitMismatch" in failed["result"]["error"]


def test_worker_termination_signal_stops_active_process_group(tmp_path, monkeypatch):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([
        Unit(
            unit_id="interrupted",
            cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
            gpus=0,
            max_attempts=2,
        )
    ])
    claim = q.claim(owner={"host": "test", "gpu": -1})
    assert claim is not None
    shutdown = threading.Event()
    shutdown.set()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(worker, "CHILD_TERMINATE_GRACE_S", 0.1)

    assert not worker.run_claim(
        q,
        claim,
        gpu=-1,
        log_dir=log_dir,
        shutdown=shutdown,
    )
    report = q.status()
    assert report["counts"]["pending"] == 1
    pending = json.loads(
        (q.root / "pending" / "interrupted.json").read_text(encoding="utf-8")
    )
    assert "WorkerTerminated" in pending["last_failure"]["error"]


def test_local_audit_recovery_releases_only_verified_local_claim(
    tmp_path,
    monkeypatch,
):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("aud__qwen25_7b__a181")])
    claim = q.claim(owner={"host": "test-host", "gpu": 0, "pid": 12345})
    assert claim is not None
    monkeypatch.setattr(recover_local_audit.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(recover_local_audit, "process_snapshot", lambda: {})

    released = recover_local_audit.recover(
        queue=q.root.resolve(),
        config=(ROOT / "configs/channel_matrix/7b_tofu.yaml").resolve(),
        model_id="qwen25_7b",
        unit_prefix="aud__qwen25_7b",
        grace_seconds=0.1,
    )

    assert released == ["aud__qwen25_7b__a181"]
    assert q.status()["counts"]["failed"] == 1


def test_local_audit_recovery_refuses_remote_claim(tmp_path, monkeypatch):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("aud__qwen25_7b__a181")])
    assert q.claim(owner={"host": "other-host", "gpu": 0, "pid": 12345})
    monkeypatch.setattr(recover_local_audit.socket, "gethostname", lambda: "test-host")

    with pytest.raises(RuntimeError, match="another host"):
        recover_local_audit.recover(
            queue=q.root.resolve(),
            config=(ROOT / "configs/channel_matrix/7b_tofu.yaml").resolve(),
            model_id="qwen25_7b",
            unit_prefix="aud__qwen25_7b",
            grace_seconds=0.1,
        )


def test_stale_claim_terminates_entire_child_process_group(
    tmp_path,
    monkeypatch,
):
    q = WorkQueue(tmp_path / "q")
    started = tmp_path / "started"
    descendant_survived = tmp_path / "descendant-survived"
    child_code = (
        "import pathlib,time;"
        f"time.sleep(1);pathlib.Path({str(descendant_survived)!r}).write_text('bad')"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(started)!r}).write_text('started');"
        "time.sleep(30)"
    )
    q.enqueue([
        Unit(
            unit_id="stale-process-group",
            cmd=[sys.executable, "-c", parent_code],
            gpus=0,
            max_attempts=1,
        )
    ])
    claim = q.claim(owner={"host": "old", "gpu": -1})
    assert claim is not None
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(worker, "HEARTBEAT_INTERVAL_S", 0.02)
    monkeypatch.setattr(worker, "CHILD_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(worker, "CHILD_TERMINATE_GRACE_S", 0.1)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.time() + 3
    while not started.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert started.is_file()

    assert q.requeue_stale(
        max_age_s=1,
        now=time.time() + 3600,
    ) == ["stale-process-group"]
    replacement = q.claim(owner={"host": "new", "gpu": -1})
    assert replacement is not None
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ClaimLostError)
    time.sleep(1.1)
    assert not descendant_survived.exists()
    q.complete(replacement, {"exit_code": 0})


def test_make_units_calibration_shards_by_author_and_objective():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    units = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "calibration", ["qwen25_7b"], 2
    )
    objectives = list(cfg["calibration"]["objective_grid"])
    authors = cfg["calibration"]["authors"]
    assert len(units) == len(authors) * len(objectives)
    assert units[0].unit_id == f"cal__qwen25_7b__a{authors[0]}__{objectives[0]}"
    for u in units:
        assert "--only-authors" in u.cmd and "--only-objectives" in u.cmd
        assert "--resume" in u.cmd and u.gpus == 1
        assert len(u.code_commit) == 40
        assert u.cmd[u.cmd.index("--only-objectives") + 1] in objectives

    audit_units = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "audit", ["qwen25_7b"], 2
    )
    assert [u.unit_id for u in audit_units] == [
        f"aud__qwen25_7b__a{a}" for a in cfg["audit"]["authors"]
    ]
    for u in audit_units:
        assert "--only-objectives" not in u.cmd


def test_cancel_moves_pending_and_claimed_units_to_failed(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    q.claim(owner={"host": "n1", "gpu": 0})  # claims a
    assert q.cancel("a") == "claimed"
    assert q.cancel("b") == "pending"
    report = q.status()
    assert report["counts"] == {"pending": 0, "claimed": 0, "done": 0, "failed": 2}
    with pytest.raises(FileNotFoundError):
        q.cancel("a")
    # cancelled units can be revived explicitly
    assert sorted(q.retry_failed()) == ["a", "b"]


def test_make_units_alpha_phases_shard_by_author_and_seed():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    dev = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "alpha-development", ["qwen25_7b"], 2
    )
    audit = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "alpha-audit", ["qwen25_7b"], 2
    )
    dev_authors = set(cfg["alpha_protection"]["development"]["authors"])
    audit_block = cfg["alpha_protection"]["audit"]
    assert len(dev) == len(dev_authors) * len(cfg["alpha_protection"]["development"]["seeds"])
    assert len(audit) == len(audit_block["authors"]) * len(audit_block["seeds"])
    for u in dev + audit:
        assert "--worker" in u.cmd and "--author" in u.cmd and "--seed" in u.cmd
    assert not {u.unit_id for u in dev} & {u.unit_id for u in audit}


def test_make_units_rejects_disabled_model():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError):
        make_units._enabled_models(cfg, {"llama31_8b"})


@pytest.mark.parametrize("scale", ["7b", "14b"])
def test_h100_tofu_configs_do_not_require_minilm(scale):
    path = ROOT / f"configs/channel_matrix/{scale}_tofu.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "sentence_encoder" not in cfg["common"]
    assert "knn_embed" not in cfg["audit"]["predictors"]


def test_replicate_units_rebuild_exact_cli_with_seed_override():
    import make_replicate_units as mru

    gate_source = (ROOT / "experiments/gate_1p5b/gate.py").read_text(encoding="utf-8")
    flags = mru.parse_gate_flags(gate_source)
    # spot-check the parser map against known gate.py flags
    assert flags["seed"]["flag"] == "--seed" and not flags["seed"]["store_true"]
    assert flags["require_sft_target"]["store_true"]
    assert flags["probe_dirs"]["flag"] == "--probe-dirs"

    manifest = {
        "model_id": "qwen25_1p5b",
        "objectives": ["npo"],
        "cli": {
            "model": "/group-volume/models/Qwen2.5-1.5B-Instruct",
            "seed": 2025,
            "probe_dirs": 64,
            "gen_lr": 2e-06,
            "require_sft_target": True,
            "smoke": False,
            "run_tag": "chanbal2",
            "out_dir": "",
            "extra_predictors": "fd_norm",
        },
    }
    units = mru.build_units(manifest, [2026, 2027], "chanbal2", "python", gate_source)
    assert [u.unit_id for u in units] == ["gate__chanbal2-s2026", "gate__chanbal2-s2027"]
    for unit, seed in zip(units, [2026, 2027]):
        cmd = unit.cmd
        assert unit.max_attempts == 1  # gate run tags are append-only
        assert cmd[cmd.index("--seed") + 1] == str(seed)
        assert cmd[cmd.index("--run-tag") + 1] == f"chanbal2-s{seed}"
        assert "--out-dir" not in cmd and "--smoke" not in cmd
        assert "--require-sft-target" in cmd
        assert cmd[cmd.index("--extra-predictors") + 1] == "fd_norm"
        assert cmd[cmd.index("--gen-lr") + 1] == "2e-06"

    # replicating the source seed itself is an error
    with pytest.raises(ValueError):
        mru.build_units(manifest, [2025], "chanbal2", "python", gate_source)

    # CLI drift (manifest key unknown to today's gate.py) must be a hard error
    drifted = {"model_id": "m", "objectives": [], "cli": {"seed": 2025, "removed_flag": 1}}
    with pytest.raises(ValueError, match="drifted"):
        mru.build_units(drifted, [2026], "t", "python", gate_source)


def test_replicate_seed_expansion():
    import make_replicate_units as mru

    assert mru.expand_seeds("2026-2029,2040") == [2026, 2027, 2028, 2029, 2040]
    with pytest.raises(ValueError):
        mru.expand_seeds("2026,2026")


def test_fleet_assignment_mismatch_detection(tmp_path):
    import fleet_status as fs

    cfg = tmp_path / "fleet.yaml"
    cfg.write_text(
        "assignments:\n"
        "  node-a: runs/cluster_queue/wave1\n"
        "  node-b: runs/cluster_queue/wave1_14b\n",
        encoding="utf-8",
    )
    assignments = fs.load_assignments(cfg)
    assert assignments == {"node-a": "runs/cluster_queue/wave1",
                           "node-b": "runs/cluster_queue/wave1_14b"}
    # node serving its own queue: fine (relative or absolute path spelling)
    assert not fs.assignment_mismatch(assignments, "node-a", "runs/cluster_queue/wave1")
    assert not fs.assignment_mismatch(
        assignments, "node-a", fs.RUNS_ROOT / "cluster_queue/wave1")
    # node grabbing another campaign's queue: flagged
    assert fs.assignment_mismatch(assignments, "node-b", "runs/cluster_queue/wave1")
    # unknown host: never flagged
    assert not fs.assignment_mismatch(assignments, "node-c", "runs/cluster_queue/wave1")
    assert fs.load_assignments(tmp_path / "missing.yaml") == {}


def test_node_watch_snapshot_and_worker_parsing(tmp_path):
    import node_watch as nw

    parsed = nw.parse_worker_cmdline(
        ["python", "-u", "experiments/cluster/worker.py",
         "--queue", "runs/cluster_queue/wave1", "--gpu", "3", "--wait"])
    assert parsed == {"queue": "runs/cluster_queue/wave1", "gpu": 3}
    assert nw.parse_worker_cmdline(["python", "train.py", "--gpu", "3"]) is None
    assert nw.parse_worker_cmdline(
        ["python", "experiments/cluster/worker.py", "--gpu", "3"]) is None

    path = nw.write_snapshot(tmp_path / "status", "test-host")
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["host"] == "test-host"
    assert isinstance(snap["gpus"], list) and isinstance(snap["workers"], list)
    assert snap["updated_epoch"] > 0


def test_queue_cli_roundtrip(tmp_path):
    units_file = tmp_path / "units.jsonl"
    units_file.write_text(
        json.dumps({"unit_id": "cli-a", "cmd": ["true"], "gpus": 0}) + "\n",
        encoding="utf-8",
    )
    queue_dir = tmp_path / "q"
    script = ROOT / "experiments/cluster/workqueue.py"
    for action, extra in [
        ("init", []),
        ("enqueue", ["--units", str(units_file)]),
        ("status", []),
        ("requeue-stale", []),
    ]:
        out = subprocess.run(
            [sys.executable, str(script), action, "--queue", str(queue_dir), *extra],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
    assert (queue_dir / "pending" / "cli-a.json").exists()


def test_enqueue_skip_existing_tops_up_after_duplicates(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("u0"), _unit("u1")])
    # Default stays append-only strict.
    try:
        q.enqueue([_unit("u1"), _unit("u2")])
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("duplicate must raise without skip_existing")
    # skip_existing adds the units AFTER the duplicate instead of aborting.
    added = q.enqueue([_unit("u1"), _unit("u3")], skip_existing=True)
    assert added == ["u3"]
    assert q.last_skipped == ["u1"]
    # u0 u1 u3 — the strict call aborted at duplicate u1, losing u2 entirely;
    # that lost-tail behavior is exactly why top-ups must pass skip_existing.
    assert q.status()["counts"]["pending"] == 3
