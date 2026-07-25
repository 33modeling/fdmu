from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "cluster"))

from move_runs import move_tree  # noqa: E402


def test_move_tree_preserves_destination_conflicts(tmp_path):
    source = tmp_path / "checkout-runs"
    target = tmp_path / "group-runs"
    (source / "same").mkdir(parents=True)
    (target / "same").mkdir(parents=True)
    (source / "same/result.json").write_text("local", encoding="utf-8")
    (target / "same/result.json").write_text("shared", encoding="utf-8")
    (source / "new.txt").write_text("new", encoding="utf-8")

    moved, _, conflicts = move_tree(source, target)

    assert moved == 2
    assert conflicts == 1
    assert not source.exists()
    assert (target / "same/result.json").read_text(encoding="utf-8") == "shared"
    conflict = list((target / "_migration_conflicts").glob("*/same/result.json"))
    assert len(conflict) == 1
    assert conflict[0].read_text(encoding="utf-8") == "local"
    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
