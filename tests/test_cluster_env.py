from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_env_separates_shared_state_from_host_scratch(tmp_path):
    group_root = tmp_path / "group-volume"
    hf_home = group_root / "data/hf_home"
    hf_home.mkdir(parents=True)
    runtime_base = group_root / "scratch"
    work_root = runtime_base / "researcher/node-7b"
    local_env = tmp_path / "cluster-local.sh"
    local_env.write_text(
        "\n".join(
            (
                f"GROUP_VOLUME_ROOT={group_root}",
                f"CLUSTER_HF_HOME={hf_home}",
                f"CLUSTER_RUNTIME_BASE={runtime_base}",
                f"CLUSTER_WORK_ROOT={tmp_path / 'wrong-local-work'}",
                f"CLUSTER_RUNS_ROOT={tmp_path / 'wrong-local-runs'}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "USER": "researcher",
        "HOSTNAME": "node-7b",
        "CLUSTER_LOCAL_ENV": str(local_env),
        "CLUSTER_RUNS_ROOT": str(tmp_path / "wrong-inherited-runs"),
        "CLUSTER_WORK_ROOT": str(tmp_path / "wrong-inherited-work"),
        "CLUSTER_TMPDIR": str(tmp_path / "wrong-inherited-tmp"),
    }
    command = (
        "set -eu; "
        "source experiments/cluster/cluster_env.sh; "
        "printf 'RUNS=%s\\nWORK=%s\\nTMP=%s\\nHF=%s\\n' "
        '"$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT" "$TMPDIR" "$HF_HOME"'
    )

    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"RUNS={ROOT / 'runs'}" in result.stdout
    assert f"WORK={work_root}" in result.stdout
    assert f"TMP={work_root / 'tmp'}" in result.stdout
    assert f"HF={hf_home}" in result.stdout
    assert not list(work_root.rglob(".fdmu-write-probe-*"))
