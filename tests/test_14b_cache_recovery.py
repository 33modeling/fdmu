"""Dependency-light regression tests for 14B cache and launcher recovery."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/gate_1p5b/gate.py"
LAUNCHER = ROOT / "experiments/cluster/run_tofu_14b_h100.sh"


class _FakeTensor:
    def __init__(self, size: int = 4, *, dtype: str = "float32"):
        self.size = size
        self.shape = (size,)
        self.dtype = dtype

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _FakeTensor(self.size, dtype=self.dtype)


class _FakeTorch:
    def __init__(self):
        self.load_error: Exception | None = None
        self.loaded_state = {"weight": _FakeTensor()}
        self.load_calls = 0
        self.save_calls: list[tuple[Path, dict]] = []
        self.save_errors: list[Exception] = []
        self.on_save = None

    def is_tensor(self, value) -> bool:
        return isinstance(value, _FakeTensor)

    def save(self, state, path, **kwargs) -> None:
        del state
        self.save_calls.append((Path(path), kwargs))
        if self.on_save is not None:
            self.on_save()
        if self.save_errors:
            raise self.save_errors.pop(0)
        Path(path).write_bytes(b"serialized-state")

    def load(self, path, **kwargs):
        del path, kwargs
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return self.loaded_state


class _FakeModel:
    def __init__(self):
        self.loaded = None
        self.load_strict = None
        self.expected_state = {"weight": _FakeTensor()}

    def load_state_dict(self, state, *, strict=True) -> None:
        self.loaded = state
        self.load_strict = strict

    def state_dict(self):
        return self.expected_state

    def named_parameters(self):
        return self.expected_state.items()


def _load_cache_functions(fake_torch: _FakeTorch) -> dict:
    wanted = {
        "_sft_cache_contract",
        "_sft_cache_guard",
        "_sha256_file",
        "_fsync_directory",
        "_sft_cache_local_stage",
        "_remove_stale_sft_temporaries_locked",
        "_quarantine_sft_cache_locked",
        "_known_sft_cache_corruption",
        "_validate_sft_result",
        "_sft_expected_state",
        "_snapshot_sft_state",
        "_validate_sft_state_dict",
        "_apply_sft_state",
        "_read_sft_cache_metadata_locked",
        "_load_sft_cache",
        "_write_sft_cache",
    }
    tree = ast.parse(GATE.read_text(encoding="utf-8"), filename=str(GATE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    namespace = {
        "Path": Path,
        "contextmanager": contextmanager,
        "datetime": datetime,
        "timezone": timezone,
        "errno": errno,
        "fcntl": fcntl,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "pickle": pickle,
        "shutil": shutil,
        "tempfile": tempfile,
        "time": time,
        "uuid": uuid,
        "torch": fake_torch,
        "ROOT": ROOT,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(GATE), "exec"), namespace)
    return namespace


class CacheRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runs = self.root / "runs"
        self.cache = self.root / "cache" / "request.pt"
        self.cache.parent.mkdir(parents=True)
        self.contract = {
            "schema": "sft-cache-v3",
            "request": "tofu-a181",
            "state_scope": "full_model",
            "state_keys": None,
        }
        self.result = {
            "steps": 1,
            "full_mean_nll": 0.5,
            "target": 0.8,
            "reached": True,
        }
        self.logs: list[str] = []
        self.torch = _FakeTorch()
        self.functions = _load_cache_functions(self.torch)
        self.env = mock.patch.dict(
            os.environ,
            {
                "CLUSTER_RUNS_ROOT": str(self.runs),
                "SFT_CACHE_LOCAL_TMPDIR": str(self.root / "local-stage"),
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tempdir.cleanup()

    @property
    def metadata_path(self) -> Path:
        return self.cache.with_suffix(".pt.json")

    def _write_metadata(self, contract=None, integrity=None) -> None:
        payload = {
            "contract": self.contract if contract is None else contract,
            "sft_result": self.result,
        }
        if integrity is not None:
            payload["integrity"] = integrity
        self.metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    def _quarantine_dirs(self) -> list[Path]:
        root = self.runs / "forensics" / "sft-cache-corrupt"
        return sorted(path for path in root.glob("*") if path.is_dir())

    def test_incomplete_final_pair_is_preserved_and_becomes_cache_miss(self):
        self.cache.write_bytes(b"partial")

        loaded = self.functions["_load_sft_cache"](
            _FakeModel(), self.cache, self.contract, self.logs.append
        )

        self.assertIsNone(loaded)
        self.assertFalse(self.cache.exists())
        quarantine = self._quarantine_dirs()
        self.assertEqual(len(quarantine), 1)
        self.assertEqual((quarantine[0] / self.cache.name).read_bytes(), b"partial")
        manifest = json.loads(
            (quarantine[0] / "quarantine.json").read_text(encoding="utf-8")
        )
        self.assertIn("incomplete final cache pair", manifest["reason"])

    def test_known_torch_corruption_is_quarantined_but_generic_io_is_not(self):
        self.cache.write_bytes(b"broken archive")
        self._write_metadata()
        self.torch.load_error = RuntimeError(
            "[enforce fail at inline_container.cc:659] "
            "unexpected pos 123456 vs 123400"
        )

        loaded = self.functions["_load_sft_cache"](
            _FakeModel(), self.cache, self.contract, self.logs.append
        )

        self.assertIsNone(loaded)
        self.assertEqual(len(self._quarantine_dirs()), 1)

        self.cache.write_bytes(b"temporarily unavailable")
        self._write_metadata()
        self.torch.load_error = OSError(5, "Input/output error")
        with self.assertRaises(OSError):
            self.functions["_load_sft_cache"](
                _FakeModel(), self.cache, self.contract, self.logs.append
            )
        self.assertTrue(self.cache.exists())
        self.assertTrue(self.metadata_path.exists())
        self.assertEqual(len(self._quarantine_dirs()), 1)

    def test_fixed_path_contract_mismatch_is_quarantined_and_rebuilt(self):
        self.cache.write_bytes(b"valid legacy cache")
        self._write_metadata(contract={"schema": "different"})

        loaded = self.functions["_load_sft_cache"](
            _FakeModel(), self.cache, self.contract, self.logs.append
        )

        self.assertIsNone(loaded)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.metadata_path.exists())
        self.assertEqual(len(self._quarantine_dirs()), 1)
        manifest = json.loads(
            (self._quarantine_dirs()[0] / "quarantine.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("contract mismatch", manifest["reason"])

    def test_writer_replaces_fixed_path_contract_mismatch_in_one_call(self):
        self.cache.write_bytes(b"old-full-model-cache")
        self._write_metadata(contract={"schema": "sft-cache-v2"})

        written = self.functions["_write_sft_cache"](
            self.cache,
            self.contract,
            self.result,
            {"weight": _FakeTensor()},
            self.logs.append,
        )

        self.assertTrue(written)
        self.assertEqual(self.cache.read_bytes(), b"serialized-state")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["contract"], self.contract)
        self.assertEqual(len(self._quarantine_dirs()), 1)

    def test_new_cache_records_integrity_and_detects_later_tampering(self):
        stale = self.cache.with_name(f".{self.cache.name}.dead-worker.tmp")
        stale.write_bytes(b"abandoned")
        old = time.time() - 25 * 60 * 60
        os.utime(stale, (old, old))
        active = self.cache.with_name(f".{self.cache.name}.active-worker.tmp")
        active.write_bytes(b"in-progress")
        local_root = self.root / "local-stage"
        local_root.mkdir()
        local_stale = local_root / f"{self.cache.name}.dead-worker.stage"
        local_stale.write_bytes(b"abandoned-local")
        os.utime(local_stale, (old, old))
        written = self.functions["_write_sft_cache"](
            self.cache,
            self.contract,
            self.result,
            {"weight": _FakeTensor()},
            self.logs.append,
        )
        self.assertTrue(written)
        self.assertFalse(stale.exists())
        self.assertFalse(local_stale.exists())
        self.assertTrue(active.exists())
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        integrity = metadata["integrity"]
        self.assertEqual(integrity["algorithm"], "sha256")
        self.assertEqual(integrity["size_bytes"], self.cache.stat().st_size)
        self.assertEqual(
            integrity["sha256"],
            hashlib.sha256(self.cache.read_bytes()).hexdigest(),
        )
        self.assertEqual(integrity["serialization"], "torch-legacy")
        self.assertEqual(
            self.torch.save_calls[0][1],
            {"_use_new_zipfile_serialization": False},
        )
        self.assertEqual(
            self.torch.save_calls[0][0].parent,
            self.root / "local-stage",
        )

        model = _FakeModel()
        loaded = self.functions["_load_sft_cache"](
            model, self.cache, self.contract, self.logs.append
        )
        self.assertEqual(loaded, self.result)
        self.assertIs(model.loaded, self.torch.loaded_state)
        self.assertTrue(model.load_strict)

        self.cache.write_bytes(b"tampered")
        loaded = self.functions["_load_sft_cache"](
            _FakeModel(), self.cache, self.contract, self.logs.append
        )
        self.assertIsNone(loaded)
        self.assertEqual(len(self._quarantine_dirs()), 1)

    def test_serialization_does_not_hold_publish_lock(self):
        real_guard = self.functions["_sft_cache_guard"]
        lock_depth = 0

        @contextmanager
        def tracked_guard(path, *, exclusive):
            nonlocal lock_depth
            with real_guard(path, exclusive=exclusive):
                lock_depth += 1
                try:
                    yield
                finally:
                    lock_depth -= 1

        self.functions["_sft_cache_guard"] = tracked_guard

        def observe_save():
            self.assertEqual(lock_depth, 0)

        self.torch.on_save = observe_save
        self.assertTrue(
            self.functions["_write_sft_cache"](
                self.cache,
                self.contract,
                self.result,
                {"weight": _FakeTensor()},
                self.logs.append,
            )
        )
        self.assertEqual(lock_depth, 0)

    def test_known_inline_container_write_failure_retries_once(self):
        self.torch.save_errors = [
            RuntimeError(
                "[enforce fail at inline_container.cc:659] "
                "unexpected pos 123456 vs 123400"
            )
        ]

        written = self.functions["_write_sft_cache"](
            self.cache,
            self.contract,
            self.result,
            {"weight": _FakeTensor()},
            self.logs.append,
        )

        self.assertTrue(written)
        self.assertEqual(len(self.torch.save_calls), 2)
        self.assertTrue(any("RETRY SFT cache serialization" in x for x in self.logs))
        self.assertTrue(self.cache.exists())
        self.assertTrue(self.metadata_path.exists())

    def test_generic_write_io_error_is_not_retried_or_hidden(self):
        self.torch.save_errors = [OSError(5, "Input/output error")]

        with self.assertRaises(OSError):
            self.functions["_write_sft_cache"](
                self.cache,
                self.contract,
                self.result,
                {"weight": _FakeTensor()},
                self.logs.append,
            )

        self.assertEqual(len(self.torch.save_calls), 1)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.metadata_path.exists())
        self.assertEqual(list((self.root / "local-stage").glob("*")), [])

    def test_probe_block_snapshot_and_load_only_selected_keys(self):
        model = _FakeModel()
        model.expected_state = {
            "base.weight": _FakeTensor(8),
            "block.weight": _FakeTensor(4),
            "block.bias": _FakeTensor(2),
        }
        contract = {
            **self.contract,
            "state_scope": "trainable_block",
            "state_keys": ["block.bias", "block.weight"],
        }

        snapshot = self.functions["_snapshot_sft_state"](model, contract)
        self.assertEqual(set(snapshot), {"block.weight", "block.bias"})
        self.assertIsNone(
            self.functions["_validate_sft_state_dict"](model, snapshot, contract)
        )

        self.functions["_apply_sft_state"](model, snapshot, contract)
        self.assertEqual(set(model.loaded), {"block.weight", "block.bias"})
        self.assertFalse(model.load_strict)

    def test_probe_block_contract_declares_only_trainable_state(self):
        args = SimpleNamespace(
            trainable_scope="probe_block",
            model="/models/qwen-14b",
            dtype="bfloat16",
            attn_impl="",
            smoke=False,
            sft_lr=1e-5,
            sft_steps=400,
            sft_target_loss=0.8,
            sft_eval_every=100,
            batch_size=4,
            seed=0,
        )
        request = SimpleNamespace(
            request_id="tofu-a181",
            universe=SimpleNamespace(sha="universe-sha"),
            forget_sha="forget-sha",
        )
        block = SimpleNamespace(pattern=r".*\.down_proj\.weight")

        contract = self.functions["_sft_cache_contract"](
            args,
            request,
            block,
            ["layers.39.mlp.down_proj.weight", "layers.38.mlp.down_proj.weight"],
        )

        self.assertEqual(contract["schema"], "sft-cache-v3")
        self.assertEqual(contract["state_scope"], "trainable_block")
        self.assertEqual(
            contract["state_keys"],
            [
                "layers.38.mlp.down_proj.weight",
                "layers.39.mlp.down_proj.weight",
            ],
        )

    def test_probe_block_cache_rejects_extra_base_model_tensor(self):
        model = _FakeModel()
        model.expected_state = {
            "base.weight": _FakeTensor(8),
            "block.weight": _FakeTensor(4),
        }
        contract = {
            **self.contract,
            "state_scope": "trainable_block",
            "state_keys": ["block.weight"],
        }
        state = {
            "base.weight": _FakeTensor(8),
            "block.weight": _FakeTensor(4),
        }

        error = self.functions["_validate_sft_state_dict"](model, state, contract)

        self.assertIn("unexpected=['base.weight']", error)

    def test_state_contract_mismatch_is_quarantined_before_model_mutation(self):
        self.cache.write_bytes(b"serialized-state")
        self._write_metadata(
            integrity={
                "algorithm": "sha256",
                "size_bytes": self.cache.stat().st_size,
                "sha256": hashlib.sha256(self.cache.read_bytes()).hexdigest(),
            }
        )
        self.torch.loaded_state = {"wrong-key": _FakeTensor()}
        model = _FakeModel()

        loaded = self.functions["_load_sft_cache"](
            model, self.cache, self.contract, self.logs.append
        )

        self.assertIsNone(loaded)
        self.assertIsNone(model.loaded)
        self.assertEqual(len(self._quarantine_dirs()), 1)
        manifest = json.loads(
            (self._quarantine_dirs()[0] / "quarantine.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("state-dict keys mismatch", manifest["reason"])

    def test_concurrent_winner_is_preserved_after_local_serialization(self):
        winner_bytes = b"winner-state"

        def publish_winner():
            self.torch.on_save = None
            self.cache.write_bytes(winner_bytes)
            self._write_metadata(
                integrity={
                    "algorithm": "sha256",
                    "size_bytes": len(winner_bytes),
                    "sha256": hashlib.sha256(winner_bytes).hexdigest(),
                }
            )

        self.torch.on_save = publish_winner
        written = self.functions["_write_sft_cache"](
            self.cache,
            self.contract,
            self.result,
            {"weight": _FakeTensor()},
            self.logs.append,
        )

        self.assertFalse(written)
        self.assertEqual(self.cache.read_bytes(), winner_bytes)
        self.assertTrue(
            any("concurrent worker" in message for message in self.logs)
        )
        self.assertEqual(list((self.root / "local-stage").glob("*")), [])

    def test_directory_fsync_allows_only_explicitly_unsupported_filesystems(self):
        with mock.patch("os.fsync", side_effect=OSError(errno.ENOTSUP, "unsupported")):
            self.functions["_fsync_directory"](self.cache.parent)

        with mock.patch("os.fsync", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaisesRegex(OSError, "I/O error"):
                self.functions["_fsync_directory"](self.cache.parent)


class LauncherContractTest(unittest.TestCase):
    def test_gpu_preflight_precedes_recovery_and_retry_is_audit_scoped(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("stage gpu-exclusive-preflight"),
            source.index("stage failed-audit-partial-quarantine"),
        )
        self.assertNotIn("stage fidelity", source)
        self.assertIn("wave1_14b must not retain GPU1-7 workers", source)
        self.assertIn("--query-compute-apps=pid,process_name,used_gpu_memory", source)
        self.assertIn('config["audit"]["authors"]', source)
        self.assertIn('RETRY_ARGS+=(--unit "$unit_id")', source)
        self.assertIn('--queue "$QUEUE" "${RETRY_ARGS[@]}"', source)


if __name__ == "__main__":
    unittest.main()
