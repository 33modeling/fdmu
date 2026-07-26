from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_4090_pipeline_accepts_resolved_calibration_boundary_only():
    launcher = (ROOT / "local_run/run_tofu_1p5b_4090x2.sh").read_text(
        encoding="utf-8"
    )
    assert 'run_stage_accept 2 calibration "0 4"' in launcher
    assert "run_stage 3 automatic-parent-freeze" in launcher
    assert "run_stage 4 joint-sweep" in launcher
    assert "run_stage 5 declared-fidelity" in launcher
    assert "run_stage 6 target-evidence-latex" in launcher
    assert 'RUN_FINALIZE:-1' in launcher
    assert "require_file_approval" not in launcher
    assert "APPROVE JOINT" not in launcher
    assert "read -r response" not in launcher
    assert "this is the only operator command" in launcher
    assert "rerun this same command" in launcher
    assert launcher.index("run_stage 5 declared-fidelity") < launcher.index(
        "run_stage 6 target-evidence-latex"
    )
    assert '"0 3 4"' not in launcher
    assert 'STATUS_FILE="$RUN_ROOT/CURRENT_STAGE.txt"' in launcher
    assert 'ERROR_FILE="$RUN_ROOT/LAST_ERROR.txt"' in launcher
    assert "PIPELINE_HEARTBEAT_SECONDS" in launcher
    assert 'setsid -- "$@" &' in launcher
    assert 'kill -TERM -- "-$ACTIVE_STAGE_PGID"' in launcher
    assert 'kill -KILL -- "-$ACTIVE_STAGE_PGID"' in launcher
