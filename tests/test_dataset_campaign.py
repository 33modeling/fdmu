"""Dependency-light contracts for the 5 x 3 dataset expansion launchers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wmdp_bio", "muse_news", "rwku", "muse_books", "pistol")
SCALES = ("1p5b", "7b", "14b")


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


materialize = _module(
    "dataset_materialize", "experiments/dataset_campaign/materialize_config.py"
)
campaign = _module(
    "dataset_campaign_runner", "experiments/channel_matrix/run_campaign.py"
)
freeze = _module(
    "dataset_auto_freeze",
    "experiments/dataset_campaign/freeze_from_calibration.py",
)
renderer = _module(
    "dataset_latex_renderer", "experiments/dataset_campaign/render_latex.py"
)


def _matrix() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/dataset_campaign/matrix.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_all_fifteen_configs_validate_and_avoid_removed_dependencies(tmp_path):
    matrix = _matrix()
    seen = set()
    for dataset in DATASETS:
        for scale in SCALES:
            config = materialize.build_config(
                matrix,
                dataset=dataset,
                scale=scale,
                model_path=tmp_path / "models" / scale,
                output_root=tmp_path / "results" / f"{dataset}-{scale}",
            )
            campaign._validate_campaign(config)
            assert config["fidelity"] is None
            assert "fidelity_certificates" not in config["audit"]
            assert "knn_embed" not in config["audit"]["predictors"]
            assert "sentence_encoder" not in config["common"]
            assert config["audit"]["objective_freeze"].startswith(str(tmp_path))
            seen.add((dataset, scale))
    assert seen == {(dataset, scale) for dataset in DATASETS for scale in SCALES}


def test_dataset_request_directory_conventions():
    assert campaign._request_dirname({"dataset": "muse_news"}, 4) == "muse-news-r004"
    assert campaign._request_dirname({"dataset": "muse_books"}, 4) == "muse-books-r004"
    assert campaign._request_dirname(
        {"dataset": "pistol", "common": {"pistol_sample": 2}}, 10
    ) == "pistol-s2-e010"


def test_automatic_freeze_labels_best_observed_fallback():
    config = {
        "campaign_id": "test-campaign",
        "models": [{"id": "model", "enabled": True}],
        "calibration": {
            "authors": [0, 1],
            "seeds": [2025],
            "selection": {"forget_recall_max": 0.1},
        },
        "audit": {"objectives": ["graddiff", "rmu"]},
    }
    recommendation = {
        "source_campaign": "test-campaign",
        "development_diagnostics": [
            {
                "model": "model",
                "objective": "graddiff",
                "setting_id": "strict",
                "setting": {"lr": 1e-6, "steps": 100},
                "n_runs": 2,
                "eligible": True,
                "forget_recall_max": 0.08,
                "mean_dnll": 1.0,
                "cvar05_dnll": 3.0,
            },
            {
                "model": "model",
                "objective": "rmu",
                "setting_id": "near",
                "setting": {"lr": 2e-6, "steps": 200},
                "n_runs": 2,
                "eligible": False,
                "forget_recall_max": 0.20,
                "mean_dnll": 0.5,
                "cvar05_dnll": 2.0,
            },
            {
                "model": "model",
                "objective": "rmu",
                "setting_id": "far",
                "setting": {"lr": 1e-6, "steps": 100},
                "n_runs": 2,
                "eligible": False,
                "forget_recall_max": 0.40,
                "mean_dnll": 0.1,
                "cvar05_dnll": 0.2,
            },
        ],
    }
    payload = freeze.build_freeze(config, recommendation)
    assert payload["status"] == "frozen"
    assert payload["unresolved"] == []
    assert payload["selection_status"]["model"]["graddiff"] == "strict_eligible"
    assert (
        payload["selection_status"]["model"]["rmu"]
        == "best_observed_ineligible"
    )
    assert payload["models"]["model"]["rmu"]["lr"] == 2e-6


def test_latex_renderer_writes_all_metric_columns():
    output = renderer.render(
        [
            {
                "predictor": "fd_norm",
                "objective": "rmu",
                "channel": "representation",
                "rho": "0.5",
                "auroc": "0.6",
                "overlap": "0.7",
                "tail_rho": "0.8",
            }
        ],
        "PISTOL",
        "Qwen2.5-7B",
    )
    assert "fd\\_norm" in output
    assert "Tail $\\rho$" in output
    assert "0.500 & 0.600 & 0.700 & 0.800" in output


def test_wrapper_matrix_is_complete_and_requires_explicit_action():
    wrappers = []
    for dataset in DATASETS:
        wrappers.append(
            ROOT / f"local_run/run_{dataset}_1p5b_4090x2.sh"
        )
        wrappers.append(
            ROOT / f"experiments/cluster/run_{dataset}_7b_h100.sh"
        )
        wrappers.append(
            ROOT / f"experiments/cluster/run_{dataset}_14b_h100.sh"
        )
    assert len(wrappers) == 15
    assert all(path.is_file() for path in wrappers)
    runner = (
        ROOT / "experiments/dataset_campaign/run.sh"
    ).read_text(encoding="utf-8")
    assert 'ACTION="${1:-}"' in runner
    assert "if (( $# != 1 )); then" in runner
    assert "all | preflight | plan | calibration | freeze | audit" in runner
    assert "parallel_phase.py" in runner
    assert "freeze_from_calibration.py" in runner
