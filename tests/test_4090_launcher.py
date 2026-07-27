import hashlib
import json
from pathlib import Path

import yaml

from experiments.paper.run_parent_calibration_4090x2 import (
    calibration_completion,
)


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
    assert '"$JOINT_ROOT/LATEST_FAILURE.txt"' in launcher
    assert "COMPLETE FAILURE REPORT" in launcher
    assert "PIPELINE_HEARTBEAT_SECONDS" in launcher
    assert 'setsid -- "$@" &' in launcher
    assert 'kill -TERM -- "-$ACTIVE_STAGE_PGID"' in launcher
    assert 'kill -KILL -- "-$ACTIVE_STAGE_PGID"' in launcher
    assert 'if run_isolated_stage_command "$@"; then' in launcher


def test_completed_calibration_marker_prevents_retraining(tmp_path):
    setting = "tofu_qwen25_1p5b"
    selection = tmp_path / "stage" / "parent_selection_inputs.jsonl"
    proposal = (
        tmp_path
        / "freeze_proposals"
        / "tofu_parent_freeze_1p5b.recommended.yaml"
    )
    selection.parent.mkdir(parents=True)
    proposal.parent.mkdir(parents=True)
    selection.write_text('{"complete": true}\n', encoding="utf-8")
    selection_hash = hashlib.sha256(selection.read_bytes()).hexdigest()
    proposal.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "contract": "tofu-pdf-v4-parent-freeze",
                "status": "draft",
                "unresolved": [],
                "development_artifacts": [
                    {
                        "path": str(selection.resolve()),
                        "sha256": selection_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "CALIBRATION_STATUS.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "setting": setting,
                "proposal": str(proposal.resolve()),
                "proposal_sha256": hashlib.sha256(
                    proposal.read_bytes()
                ).hexdigest(),
                "selection_input": str(selection.resolve()),
                "selection_input_sha256": selection_hash,
                "unresolved": [],
                "approval_ready": True,
            }
        ),
        encoding="utf-8",
    )

    complete, _reason = calibration_completion(tmp_path, setting)
    assert complete

    selection.write_text('{"changed": true}\n', encoding="utf-8")
    complete, reason = calibration_completion(tmp_path, setting)
    assert not complete
    assert "SHA-256" in reason
