from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.paper import summarize_joint_sweep_live as live


def _arm(name, mean, cvar, *, draw=None):
    return {
        "arm": name,
        "draw_id": draw,
        "metrics": {"feasible": True},
        "mean_damage": mean,
        "cvar95_damage": cvar,
    }


def test_live_snapshot_is_read_only_and_summarizes_partial_trial(tmp_path):
    joint = tmp_path / "joint"
    trial = joint / "trials" / "trial-a--abc"
    unit_dir = trial / "units" / "graddiff__tofu-a184__seed-2025"
    log_dir = trial / "logs" / "units" / unit_dir.name
    unit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump(
            {"execution": {"repeated_random_draws": ["rand-000", "rand-001"]}}
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "paths": {"campaign": str(campaign)},
                "gpus": [0, 1],
                "stop": {
                    "mean_damage_margin": 0.0,
                    "cvar95_damage_margin": 0.0,
                },
                "trials": [{"id": "trial-a"}],
            }
        ),
        encoding="utf-8",
    )
    (trial / "manifest.yaml").write_text(
        yaml.safe_dump({"units": [{"id": "a"}, {"id": "b"}]}),
        encoding="utf-8",
    )
    (log_dir / "attempt-001.json").write_text(
        json.dumps(
            {
                "unit": unit_dir.name,
                "attempt": 1,
                "returncode": 0,
                "duration_seconds": 100.0,
            }
        ),
        encoding="utf-8",
    )
    diagnostic = {
        "parent": "graddiff",
        "request": "tofu-a184",
        "seed": 2025,
        "arms": [
            _arm("joint", 1.0, 2.0),
            _arm("no_repair", 1.3, 2.3),
            _arm("s0", 1.4, 2.4),
            _arm("s1", 1.5, 2.5),
            _arm("repeated_random", 1.6, 2.6, draw="rand-000"),
            _arm("repeated_random", 1.7, 2.7, draw="rand-001"),
        ],
    }
    diagnostic_path = unit_dir / "protection_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
    (joint / "events.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event": "trial_started",
                    "trial_id": "trial-a",
                    "trial_dir": str(trial),
                },
                {"event": "unit_started", "unit": unit_dir.name},
                {"event": "unit_finished", "unit": unit_dir.name},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    before = diagnostic_path.read_bytes()

    snapshot = live.build_snapshot(joint, spec)
    live.write_snapshot(joint, snapshot)

    current = snapshot["current_trial"]
    assert current["completed_units"] == 1
    assert current["total_units"] == 2
    assert current["trial_eta_seconds_at_observed_rate"] == 100.0
    assert current["partial_result"]["passing_cells"] == 1
    assert current["partial_result"]["descriptive_only"] is True
    assert diagnostic_path.read_bytes() == before
    assert (joint / "live/LIVE_STATUS.json").is_file()
    assert (joint / "live/LIVE_STATUS.md").is_file()
