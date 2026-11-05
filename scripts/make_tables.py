#!/usr/bin/env python3
"""Aggregate every run's metrics.json into the T2 headline table (evaluation.md §7).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

    uv run scripts/make_tables.py [--runs-dir reports/runs] [--out reports/tables]

Globs ``<runs-dir>/**/metrics.json``, flattens each to one row per run x set x score
(``metrics_long.csv``), and writes ``t2_headline.md`` / ``t2_headline.csv``: per method, the
mean ± sd over runs of macro-F1, ECE (test_known), AUROC and FPR@95 (``--unknown-set``, default
test_unknown) and AURC (test_known), each for the method's primary score in ``PRIMARY_SCORE``.
No significance tests or bootstrap intervals here (``uaar.eval.stats``, Robert).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uaar.models.factory import PRIMARY_SCORE  # the one score per method reported here

METHOD_ORDER = ("logreg", "softmax", "mcdropout", "edl")
HEADLINE: tuple[tuple[str, str, str], ...] = (
    # (column, where, metric key); where = "known" or "unknown"
    ("macro_f1", "known", "macro_f1"),
    ("auroc", "unknown", "auroc"),
    ("fpr95", "unknown", "fpr95"),
    ("ece", "known", "ece"),
    ("aurc", "known", "aurc"),
)
RUN_KEYS = ("run_id", "method", "dataset", "protocol", "split_id", "seed", "K")


def flatten(m: dict) -> list[dict]:
    """One row per set x score of a metrics.json dict; score-free metrics repeat on each row."""
    head = {k: m.get(k) for k in RUN_KEYS}
    rows = []
    for set_name, block in m.items():
        if not isinstance(block, dict) or "scores" not in block:
            continue
        shared = {"n": block.get("n")}
        shared |= {k: v for k, v in (block.get("closed_set") or {}).items() if k != "n"}
        shared |= block.get("calibration") or {}
        for score, vals in block["scores"].items():
            rows.append(head | {"set": set_name, "score": score} | shared | vals)
    return rows


def headline(long: pd.DataFrame, unknown_set: str) -> pd.DataFrame:
    """Mean and sd over runs per method for the HEADLINE metrics at each primary score."""
    long = long[long["score"] == long["method"].map(PRIMARY_SCORE)]
    out = []
    methods = sorted(
        long["method"].unique(),
        key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m),
    )
    for method in methods:
        d = long[long["method"] == method]
        row: dict = {"method": method, "score": PRIMARY_SCORE.get(method)}
        row["n_runs"] = int(d["run_id"].nunique())
        for col, where, key in HEADLINE:
            sub = d[d["set"] == ("test_known" if where == "known" else unknown_set)]
            v = pd.to_numeric(sub[key], errors="coerce") if key in sub else pd.Series([], float)
            v = v.dropna()
            row[f"{col}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{col}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(t: pd.DataFrame, unknown_set: str) -> str:
    """Markdown table with ``mean ± sd`` cells (sd omitted for a single run)."""
    names = {"macro_f1": "macro-F1 ↑", "auroc": "AUROC ↑", "fpr95": "FPR@95 ↓", "ece": "ECE ↓"}
    names["aurc"] = "AURC ↓"
    cols = ["Method", "Score", "Runs", *(names[c] for c, _, _ in HEADLINE)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = [r["method"], str(r["score"]), str(r["n_runs"])]
        for c, _, _ in HEADLINE:
            m, s = r[f"{c}_mean"], r[f"{c}_sd"]
            if not np.isfinite(m):
                cells.append("–")
            else:
                cells.append(f"{m:.3f}" + (f" ± {s:.3f}" if np.isfinite(s) else ""))
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        f"\nMean ± sd over runs. AUROC and FPR@95 on `{unknown_set}` against `test_known`; "
        "macro-F1, ECE and AURC on `test_known`. Primary score per method: "
        + ", ".join(f"{k} = {v}" for k, v in PRIMARY_SCORE.items())
        + ".\n"
    )
    return "\n".join(lines) + "\n" + note


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    ap.add_argument("--out", type=Path, default=Path("reports/tables"))
    ap.add_argument("--unknown-set", default="test_unknown", help="set for AUROC and FPR@95")
    args = ap.parse_args()

    files = sorted(args.runs_dir.glob("**/metrics.json"))
    if not files:
        print(f"no metrics.json under {args.runs_dir}")
        return 1
    rows = [r for f in files for r in flatten(json.loads(f.read_text()))]
    long = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out / "metrics_long.csv", index=False)

    t2 = headline(long, args.unknown_set)
    t2.to_csv(args.out / "t2_headline.csv", index=False)
    md = to_markdown(t2, args.unknown_set)
    (args.out / "t2_headline.md").write_text(md)
    print(f"{len(files)} runs, {len(long)} rows -> {args.out}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
