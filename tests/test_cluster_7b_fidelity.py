"""Dependency-light tests for the cluster 7B fidelity publication contract."""
from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


campaign = _module(
    "cluster_7b_fidelity_campaign",
    "experiments/channel_matrix/run_campaign.py",
)


def _hold_fidelity_lock(
    certificate: str,
    entered,
    release,
) -> None:
    with campaign._exclusive_fidelity_build_lock(Path(certificate)):
        entered.set()
        release.wait(timeout=5)


def _atomic_helpers() -> dict:
    source = (ROOT / "experiments/diag/fd_fidelity.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_atomic_write_csv", "_atomic_write_json"}
    ]
    namespace = {
        "Path": Path,
        "csv": csv,
        "json": json,
        "os": os,
        "tempfile": tempfile,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT), "exec"), namespace)
    return namespace


class Cluster7BFidelityTest(unittest.TestCase):
    def setUp(self):
        self.live_config_path = ROOT / "configs/channel_matrix/7b_tofu.yaml"
        self.cfg = yaml.safe_load(
            self.live_config_path.read_text(encoding="utf-8")
        )
        self.model = campaign._enabled_models(self.cfg, {"qwen25_7b"})[0]

    def _certificate(self, **overrides) -> dict:
        common = self.cfg["common"]
        payload = {
            "schema": "fd-fidelity-certificate-v1",
            "passed": True,
            "model": str(self.model["path"]),
            "dtype": str(common["dtype"]),
            "candidate_authors": sorted(
                campaign._expand_int_ranges(
                    common["candidate_author_pools"]["calibration"]
                )
            ),
            "n_candidates": int(self.cfg["fidelity"]["n_candidates"]),
            "block_last_n": int(common["block_last_n"]),
            "R": int(common["probe_dirs"]),
            "eta": float(common["probe_norm_eta"]),
            "probe_seed": int(common["probe_seed"]),
        }
        payload.update(overrides)
        return payload

    def test_full_validator_rejects_stale_passed_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = copy.deepcopy(self.cfg)
            certificate = root / "qwen25_7b.json"
            cfg["audit"]["fidelity_certificates"]["qwen25_7b"] = str(certificate)

            certificate.write_text(
                json.dumps(self._certificate(dtype="bfloat16")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mismatch.*dtype"):
                campaign.validate_fidelity_certificate(cfg, self.model)

            certificate.write_text(
                json.dumps(self._certificate()),
                encoding="utf-8",
            )
            validated = campaign.validate_fidelity_certificate(cfg, self.model)
            self.assertEqual(validated["path"], str(certificate))
            self.assertEqual(validated["payload"]["passed"], True)

    def test_resume_preserves_invalid_pair_and_rebuilds_inside_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            output = runs / "channel_matrix_7b"
            model_dir = root / "model"
            model_dir.mkdir()
            cfg = copy.deepcopy(self.cfg)
            cfg["output_root"] = str(output)
            cfg["models"] = [
                {"id": "qwen25_7b", "path": str(model_dir), "enabled": True}
            ]
            self.model = cfg["models"][0]
            certificate = output / "fidelity" / "qwen25_7b.json"
            csv_path = output / "fidelity" / "qwen25_7b.csv"
            cfg["audit"]["fidelity_certificates"]["qwen25_7b"] = str(certificate)
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("old,csv\n", encoding="utf-8")
            stale = self._certificate(dtype="bfloat16")
            stale["model"] = str(model_dir)
            certificate.write_text(json.dumps(stale), encoding="utf-8")

            rebuilt = self._certificate()
            rebuilt["model"] = str(model_dir)

            def fake_run(cmd, dry_run, env):
                self.assertFalse(dry_run)
                Path(cmd[cmd.index("--out") + 1]).write_text(
                    "new,csv\n", encoding="utf-8"
                )
                Path(cmd[cmd.index("--certificate") + 1]).write_text(
                    json.dumps(rebuilt), encoding="utf-8"
                )

            argv = [
                "run_campaign.py",
                "--config",
                str(root / "campaign.yaml"),
                "--phase",
                "fidelity",
                "--resume",
                "--model-id",
                "qwen25_7b",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(campaign, "_load_yaml", return_value=cfg),
                mock.patch.object(campaign, "_validate_campaign"),
                mock.patch.object(
                    campaign,
                    "_git_state",
                    return_value={"code_commit": "test", "code_dirty": False},
                ),
                mock.patch.object(campaign, "_run", side_effect=fake_run),
                mock.patch.dict(
                    campaign.os.environ,
                    {"CLUSTER_RUNS_ROOT": str(runs)},
                    clear=False,
                ),
            ):
                campaign.main()

            campaign.validate_fidelity_artifact_pair(cfg, self.model)
            preserved = list(
                (runs / "forensics" / "fidelity-artifacts").glob(
                    "*/qwen25_7b.json"
                )
            )
            self.assertEqual(len(preserved), 1)
            old_payload = json.loads(preserved[0].read_text(encoding="utf-8"))
            self.assertEqual(old_payload["dtype"], "bfloat16")
            self.assertTrue(
                certificate.with_name(f"{certificate.name}.lock").is_file()
            )

    def test_exclusive_lock_serializes_contenders(self):
        with tempfile.TemporaryDirectory() as directory:
            certificate = Path(directory) / "certificate.json"
            context = multiprocessing.get_context("fork")
            first_entered = context.Event()
            release_first = context.Event()
            second_entered = context.Event()
            release_second = context.Event()
            one = context.Process(
                target=_hold_fidelity_lock,
                args=(str(certificate), first_entered, release_first),
            )
            two = context.Process(
                target=_hold_fidelity_lock,
                args=(str(certificate), second_entered, release_second),
            )
            one.start()
            self.assertTrue(first_entered.wait(timeout=2))
            two.start()
            time.sleep(0.1)
            self.assertFalse(second_entered.is_set())
            release_first.set()
            self.assertTrue(second_entered.wait(timeout=2))
            release_second.set()
            one.join(timeout=2)
            two.join(timeout=2)
            self.assertEqual(one.exitcode, 0)
            self.assertEqual(two.exitcode, 0)
            self.assertTrue(second_entered.is_set())

    def test_atomic_publish_leaves_no_temporary_files(self):
        helpers = _atomic_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "result.csv"
            json_path = root / "certificate.json"
            helpers["_atomic_write_csv"](csv_path, [{"a": 1, "b": 2}])
            helpers["_atomic_write_json"](json_path, {"passed": True})

            self.assertEqual(csv_path.read_text(encoding="utf-8"), "a,b\n1,2\n")
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"passed": True},
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_shell_orchestration_uses_shared_validator_and_targeted_retry(self):
        launcher = (
            ROOT / "experiments/cluster/run_tofu_7b_h100.sh"
        ).read_text(encoding="utf-8")
        enqueue = (
            ROOT / "experiments/cluster/enqueue_table12.sh"
        ).read_text(encoding="utf-8")
        recovery = launcher.split(
            "stage failed-audit-recovery", 1
        )[1].split("stage fidelity-contract-validation", 1)[0]

        self.assertIn("validate_fidelity_artifact_pair", launcher)
        self.assertIn("validate_fidelity_artifact_pair", enqueue)
        for author in (181, 186, 191):
            self.assertIn(f"--unit aud__qwen25_7b__a{author}", recovery)
        self.assertEqual(recovery.count("--unit aud__qwen25_7b__"), 3)


if __name__ == "__main__":
    unittest.main()
