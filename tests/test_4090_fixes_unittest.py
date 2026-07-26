from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from experiments.paper.build_evidence import (
    _fidelity_paths,
    _load_fidelity_inputs,
)
from rsus.evidence.schemas import EvidenceValidationError


ROOT = Path(__file__).resolve().parents[1]


class LauncherContractTests(unittest.TestCase):
    def test_yaml_compatibility_entrypoint_cannot_create_partial_venv(self):
        ensure = (ROOT / "local_run/ensure_4090_yaml.sh").read_text(
            encoding="utf-8"
        )
        bootstrap = (ROOT / "local_run/bootstrap_4090_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('exec bash "$ROOT/local_run/bootstrap_4090_env.sh"', ensure)
        self.assertNotIn("-m venv", ensure)
        self.assertIn('TORCH_VERSION="2.7.1"', bootstrap)
        self.assertIn('torch.__version__.split("+", 1)[0]', bootstrap)
        self.assertIn('"torch==$TORCH_VERSION"', bootstrap)

    def test_all_4090_stages_share_run_root_and_bootstrap(self):
        top = (ROOT / "local_run/run_tofu_1p5b_4090x2.sh").read_text(
            encoding="utf-8"
        )
        calibration = (
            ROOT / "local_run/run_tofu_1p5b_calibration.sh"
        ).read_text(encoding="utf-8")
        sweep = (
            ROOT / "local_run/sweep_joint_1p5b_4090x2.sh"
        ).read_text(encoding="utf-8")
        finalize = (
            ROOT / "local_run/finalize_joint_sweep_to_latex.sh"
        ).read_text(encoding="utf-8")
        for script in (top, calibration, sweep, finalize):
            self.assertIn("RUN_ROOT=", script)
            self.assertIn("bootstrap_4090_env.sh", script)
        self.assertIn('RESULTS_ROOT="${RESULTS_ROOT:-$RUN_ROOT/joint_sweep}"', top)
        self.assertIn('JOINT_ROOT="${JOINT_ROOT:-$RESULTS_ROOT}"', top)
        self.assertIn('JOINT_ROOT="${JOINT_ROOT:-$RESULTS_ROOT}"', finalize)
        self.assertIn('FINAL_ROOT="${FINAL_ROOT:-$RUN_ROOT/final}"', finalize)
        self.assertIn('RUN_FINALIZE:-1', top)
        self.assertIn(
            "bash local_run/finalize_joint_sweep_to_latex.sh", top
        )
        self.assertIn(
            "run_stage declared-fidelity bash local_run/run_tofu_1p5b_fidelity.sh",
            top,
        )
        self.assertLess(
            top.index("run_stage declared-fidelity"),
            top.index("run_stage target-evidence-latex"),
        )
        self.assertNotIn("require_file_approval", top)
        self.assertNotIn("APPROVE JOINT", top)
        self.assertNotIn("read -r response", top)
        self.assertIn(
            'PARENT_FREEZE="${PARENT_FREEZE:-$CALIBRATION_ROOT/freeze/',
            top,
        )
        approval = (
            ROOT / "local_run/approve_tofu_1p5b_parent_freeze.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--out "$PARENT_FREEZE"', approval)
        self.assertIn('--campaign "$CAMPAIGN"', approval)
        self.assertIn('--runtime "$RUNTIME"', approval)
        self.assertNotIn('token="APPROVE PARENT', approval)
        self.assertNotIn("read -r response", approval)
        self.assertIn("--approve", approval)
        self.assertIn('--parent-freeze "$PARENT_FREEZE"', sweep)
        self.assertIn('PIPELINE_LOG="$RUN_ROOT/launcher_logs/', top)
        self.assertNotIn('pip install "torch==', calibration)
        self.assertNotIn('pip install "torch==', sweep)

    def test_declared_fidelity_is_required_before_finalization(self):
        shell = (
            ROOT / "local_run/finalize_joint_sweep_to_latex.sh"
        ).read_text(encoding="utf-8")
        finalizer = (
            ROOT / "experiments/paper/finalize_joint_sweep.py"
        ).read_text(encoding="utf-8")
        fidelity = (
            ROOT / "local_run/run_tofu_1p5b_fidelity.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'FIDELITY_SUMMARY="${FIDELITY_SUMMARY:-$RUN_ROOT/fidelity/fidelity_summary.json}"',
            shell,
        )
        self.assertIn('"declared_setting_fidelity"', finalizer)
        self.assertIn(
            'summary.get("support") != "declared_setting_fidelity"',
            fidelity,
        )


class FidelityInputTests(unittest.TestCase):
    def test_override_is_limited_to_predeclared_setting_and_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "setting": "tofu_qwen25_1p5b",
                        "support": "declared_setting_fidelity",
                    }
                ),
                encoding="utf-8",
            )
            contract = SimpleNamespace(
                fidelity_inputs={
                    "tofu_qwen25_1p5b": str(root / "default.json")
                }
            )
            paths = _fidelity_paths(
                contract, [f"tofu_qwen25_1p5b={summary}"]
            )
            self.assertEqual(
                paths["tofu_qwen25_1p5b"], (summary, True)
            )
            loaded = _load_fidelity_inputs(
                contract, [f"tofu_qwen25_1p5b={summary}"]
            )
            self.assertEqual(
                loaded["tofu_qwen25_1p5b"]["setting"],
                "tofu_qwen25_1p5b",
            )
            summary.write_text(
                json.dumps({"setting": "tofu_qwen25_1p5b"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EvidenceValidationError, "declared setting-level"
            ):
                _load_fidelity_inputs(
                    contract, [f"tofu_qwen25_1p5b={summary}"]
                )
            summary.write_text(
                json.dumps(
                    {
                        "setting": "tofu_qwen25_1p5b",
                        "support": "declared_setting_fidelity",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EvidenceValidationError, "not predeclared"
            ):
                _fidelity_paths(contract, [f"other={summary}"])
            with self.assertRaisesRegex(
                EvidenceValidationError, "duplicated"
            ):
                _fidelity_paths(
                    contract,
                    [
                        f"tofu_qwen25_1p5b={summary}",
                        f"tofu_qwen25_1p5b={summary}",
                    ],
                )
            with self.assertRaisesRegex(
                EvidenceValidationError, "explicit fidelity input is missing"
            ):
                _load_fidelity_inputs(
                    contract,
                    [f"tofu_qwen25_1p5b={root / 'missing.json'}"],
                )
if __name__ == "__main__":
    unittest.main()
