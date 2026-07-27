#!/usr/bin/env python3
"""Render a diagnostic channel-matrix LaTeX table from an aggregate."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys


def _escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def render(rows: list[dict[str, str]], dataset: str, model: str) -> str:
    if not rows:
        raise ValueError("aggregate CSV has no rows")
    label_slug = re.sub(r"[^a-z0-9]+", "-", dataset.casefold()).strip("-")
    body = [
        "% Auto-generated diagnostic table; not a PDF-v4 claim table.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Dataset-expansion channel diagnostic for "
            + _escape(dataset)
            + " / "
            + _escape(model)
            + ".}"
        ),
        r"\label{tab:dataset-" + label_slug + "}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Predictor & Objective & Channel & $\rho$ & AUROC & Overlap & Tail $\rho$ \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(
            "{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                _escape(row["predictor"]),
                _escape(row["objective"]),
                _escape(row["channel"]),
                float(row["rho"]),
                float(row["auroc"]),
                float(row["overlap"]),
                float(row["tail_rho"]),
            )
        )
    body.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        output = render(rows, args.dataset, args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(output, encoding="utf-8")
        temporary.replace(args.out)
        print(f"[LATEX] rows={len(rows)} path={args.out.resolve()}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(
            f"dataset LaTeX render failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
