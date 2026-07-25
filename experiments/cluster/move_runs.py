"""Move a checkout-local runs tree into shared storage without overwriting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil


def move_tree(source: Path, target: Path) -> tuple[int, int, int]:
    source = source.resolve()
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    conflict_root = target / "_migration_conflicts" / stamp
    moved = conflicts = moved_bytes = 0

    entries = list(source.rglob("*"))
    files = [
        path
        for path in entries
        if path.is_symlink() or not path.is_dir()
    ]
    for path in files:
        relative = path.relative_to(source)
        destination = target / relative
        if os.path.lexists(destination):
            destination = conflict_root / relative
            conflicts += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = path.lstat().st_size
        shutil.move(str(path), str(destination))
        moved += 1
        moved_bytes += size
        if moved % 100 == 0:
            print(
                f"moved={moved} bytes={moved_bytes} conflicts={conflicts}",
                flush=True,
            )

    directories = sorted(
        (path for path in entries if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    source.rmdir()
    return moved, moved_bytes, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    moved, moved_bytes, conflicts = move_tree(args.source, args.target)
    print(
        f"migration complete: moved={moved} bytes={moved_bytes} "
        f"conflicts_preserved={conflicts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
