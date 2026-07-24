#!/usr/bin/env python
"""Summarize one gate run (table1.json + table2.json) into markdown and copy
the raw artifacts to the results dir. Usage: summarize.py NAME RUN_DIR RESULTS_DIR"""
import json
import shutil
import sys
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fnum(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    name, run_dir, results_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    dst = results_dir / name
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("table1.json", "table2.json", "gate.log", "seal_ledger.jsonl", "run_manifest.json"):
        src = run_dir / fn
        if src.exists():
            shutil.copy2(src, dst / fn)

    t1 = load(run_dir / "table1.json")
    t2 = load(run_dir / "table2.json")

    manifest = load(run_dir / "run_manifest.json") or {}
    model_path = manifest.get("model", "")
    model_name = model_path.rsplit("/", 1)[-1] if model_path else name
    dtype = manifest.get("dtype", "?")
    info = manifest.get("model_info") or {}
    total = info.get("total_parameters")
    params = f"{total/1e9:.2f}B params" if total else ""
    arch = info.get("architecture", "")
    header = f"\n## {model_name}  ({dtype}"
    if params:
        header += f", {params}"
    if arch:
        header += f", {arch}"
    header += f")  — label `{name}`, dir `{run_dir}`\n"
    out = [header]

    if t1:
        # detect generator rho columns; drop columns that are entirely n/a
        any_row = next(iter(t1.values()))
        gens = [k[:-4] for k in any_row if k.endswith("_rho")]
        cols = [(f"{g} rho", f"{g}_rho") for g in gens] + [("AUROC", "auroc")]

        def cell(row, key):  # extract mean float or None
            v = (row.get(key) or {}).get("mean")
            return None if v is None else float(v)

        # keep only columns with at least one real value
        cols = [(lab, key) for (lab, key) in cols
                if any(cell(r, key) is not None for r in t1.values())]
        # best (max) per column = 1st place; "good" thresholds for bolding
        col_max = {key: max(v for r in t1.values() if (v := cell(r, key)) is not None)
                   for _, key in cols}
        good_thr = {"auroc": 0.65}  # rho columns use 0.30

        def fmt(key, v):
            if v is None:
                return "n/a"
            s = f"{v:.3f}"
            if v == col_max[key]:
                return f"<u>**{s}**</u>"           # 1st place: bold + underline
            thr = good_thr.get(key, 0.30)
            return f"**{s}**" if v >= thr else s     # good number: bold

        head = "| predictor | " + " | ".join(lab for lab, _ in cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        out += ["### Table 1 — probe ranking quality (audit fold)",
                "_bold = good; <u>underline</u> = column best (1st)_", "", head, sep]
        for pred, r in t1.items():
            row = " | ".join(fmt(key, cell(r, key)) for _, key in cols)
            out.append(f"| {pred} | {row} |")
        out.append("")
    else:
        out.append("_table1.json missing_\n")

    if t2:
        out += ["### Table 2 — protection (audit dNLL; lower mean/CVaR = less collateral)",
                "| method | reached | step | audit mean dNLL | CVaR | para_recall |",
                "|---|---|---|---|---|---|"]
        for method, o in t2.items():
            if o.get("error"):
                out.append(f"| {method} | ERROR | | {o['error']} | | |")
                continue
            extra = o.get("extra") or {}
            out.append(
                f"| {method} | {o.get('reached')} | {o.get('step')} | "
                f"{fnum(o.get('native_mean'))} | {fnum(o.get('native_cvar'))} | "
                f"{fnum(extra.get('para_recall'))} |"
            )
        out.append("")
    else:
        out.append("_table2.json missing_\n")

    print("\n".join(out))


if __name__ == "__main__":
    main()
