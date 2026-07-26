"""Dependency-light regression tests for 14B cache and launcher recovery."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/gate_1p5b/gate.py"
LAUNCHER = ROOT / "experiments/cluster/run_tofu_14b_h100.sh"


class _FakeTensor:
    def __init__(self, size: int = 4):
        self.size = size

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


class _FakeTorch:
    def __init__(self):
        self.load_error: Exception | None = None
        self.loaded_state = {"weight": _FakeTensor()}
        self.load_calls = 0

    def is_tensor(self, value) -> bool:
        return isinstance(value, _FakeTensor)

    def save(self, state, path, **kwargs) -> None:
        del state, kwargs
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

    def load_state_dict(self, state) -> None:
        self.loaded = state


def _load_cache_functions(fake_torch: _FakeTorch) -> dict:
    wanted = {
        "_sft_cache_guard",
        "_sha256_file",
        "_quarantine_sft_cache_locked",
        "_known_sft_cache_corruption",
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
        "fcntl": fcntl,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "pickle": pickle,
        "shutil": shutil,
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
        self.contract = {"schema": "sft-cache-v2", "request": "tofu-a181"}
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
            {"CLUSTER_RUNS_ROOT": str(self.runs)},
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

    def test_contract_mismatch_is_not_quarantined(self):
        self.cache.write_bytes(b"valid legacy cache")
        self._write_metadata(contract={"schema": "different"})

        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            self.functions["_load_sft_cache"](
                _FakeModel(), self.cache, self.contract, self.logs.append
            )

        self.assertTrue(self.cache.exists())
        self.assertTrue(self.metadata_path.exists())
        self.assertEqual(self._quarantine_dirs(), [])

    def test_new_cache_records_integrity_and_detects_later_tampering(self):
        stale = self.cache.with_name(f".{self.cache.name}.dead-worker.tmp")
        stale.write_bytes(b"abandoned")
        written = self.functions["_write_sft_cache"](
            self.cache,
            self.contract,
            self.result,
            {"weight": _FakeTensor()},
            self.logs.append,
        )
        self.assertTrue(written)
        self.assertFalse(stale.exists())
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        integrity = metadata["integrity"]
        self.assertEqual(integrity["algorithm"], "sha256")
        self.assertEqual(integrity["size_bytes"], self.cache.stat().st_size)
        self.assertEqual(
            integrity["sha256"],
            hashlib.sha256(self.cache.read_bytes()).hexdigest(),
        )

        model = _FakeModel()
        loaded = self.functions["_load_sft_cache"](
            model, self.cache, self.contract, self.logs.append
        )
        self.assertEqual(loaded, self.result)
        self.assertIs(model.loaded, self.torch.loaded_state)

        self.cache.write_bytes(b"tampered")
        loaded = self.functions["_load_sft_cache"](
            _FakeModel(), self.cache, self.contract, self.logs.append
        )
        self.assertIsNone(loaded)
        self.assertEqual(len(self._quarantine_dirs()), 1)


class LauncherContractTest(unittest.TestCase):
    def test_gpu_preflight_precedes_fidelity_and_retry_is_audit_scoped(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("stage gpu-exclusive-preflight"),
            source.index("stage fidelity"),
        )
        self.assertIn("wave1_14b must not retain GPU1-7 workers", source)
        self.assertIn("--query-compute-apps=pid,process_name,used_gpu_memory", source)
        self.assertIn('config["audit"]["authors"]', source)
        self.assertIn('RETRY_ARGS+=(--unit "$unit_id")', source)
        self.assertIn('--queue "$QUEUE" "${RETRY_ARGS[@]}"', source)


if __name__ == "__main__":
    unittest.main()
