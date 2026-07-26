from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_4090_pipeline_accepts_resolved_calibration_boundary_only():
    launcher = (ROOT / "local_run/run_tofu_1p5b_4090x2.sh").read_text(
        encoding="utf-8"
    )
    assert 'run_stage_accept calibration "0 4"' in launcher
    assert 'run_stage parent-freeze-validation' in launcher
    assert 'run_stage joint-sweep' in launcher
    assert 'run_stage declared-fidelity' in launcher
    assert 'run_stage target-evidence-latex' in launcher
    assert 'RUN_FINALIZE:-1' in launcher
    assert "require_file_approval" not in launcher
    assert "APPROVE JOINT" not in launcher
    assert "read -r response" not in launcher
    assert launcher.index("run_stage declared-fidelity") < launcher.index(
        "run_stage target-evidence-latex"
    )
    assert '"0 3 4"' not in launcher
