#!/usr/bin/env python3
"""Aggregate every run's metrics.json into the result tables (evaluation.md §7).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

    uv run scripts/make_tables.py [--runs-dir reports/runs] [--out reports/tables]

Reads ``<runs-dir>/<run_id>/metrics.json`` (direct children only, so runs set aside in a
subfolder such as ``smoke/`` or ``edl-debug/`` stay out of the tables) and writes:

``metrics_long.csv``   one row per run x set x score.
``t2_headline.md/.csv``  per protocol and method, the mean ± sd over runs of macro-F1, ECE and
    AURC (test_known), AUROC, FPR@95 and the share accepted at tau (``--unknown-set``, default
    test_unknown), each for the method's primary score (``uaar.models.factory.primary_score``).
``scores_by_protocol.md/.csv``  per protocol, method and score: known accuracy, macro-F1, ECE,
    the share of known test windows kept, and for every unknown set the share accepted at tau,
    FPR@95 and the AUROC. This is the table of the three tiers (k4plain, k4taught, k4other).

Runs of different protocols are never averaged together. No significance tests or bootstrap
intervals are computed: the tables give the mean and standard deviation over runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uaar.models.factory import primary_score  # the one score per method reported here

METHOD_ORDER = ("logreg", "softmax", "mcdropout", "edl")
PROTOCOL_ORDER = ("p1", "k4plain", "k4taught", "k4other")
HEADLINE: tuple[tuple[str, str, str], ...] = (
    # (column, where, metric key); where = "known" or "unknown"
    ("macro_f1", "known", "macro_f1"),
    ("auroc", "unknown", "auroc"),
    ("fpr95", "unknown", "fpr95"),
    ("accepted", "unknown", "accepted"),
    ("ece", "known", "ece"),
    ("aurc", "known", "aurc"),
)
UNKNOWN_SETS = ("test_unknown", "test_nonex", "test_background", "test_lookalike")
RUN_KEYS = ("run_id", "method", "dataset", "protocol", "background", "split_id", "seed", "K")


def _order(values, reference) -> list:
    return sorted(values, key=lambda v: (reference.index(v) if v in reference else 99, str(v)))


def flatten(m: dict) -> list[dict]:
    """One row per set x score of a metrics.json dict; score-free metrics repeat on each row."""
    head = {k: m.get(k) for k in RUN_KEYS}
    head["background"] = head["background"] or "none"  # runs evaluated before the key existed
    rows = []
    for set_name, block in m.items():
        if not isinstance(block, dict) or "scores" not in block:
            continue
        shared = {"n": block.get("n")}
        shared |= {k: v for k, v in (block.get("closed_set") or {}).items() if k != "n"}
        shared |= block.get("calibration") or {}
        for score, vals in block["scores"].items():
            row = head | {"set": set_name, "score": score} | shared | vals
            if "accepted" not in row and row.get("unknown_recall") is not None:
                row["accepted"] = 1.0 - row["unknown_recall"]  # same quantity, older files
            rows.append(row)
    return rows


def _mean_sd(values) -> tuple[float, float]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return (
        float(v.mean()) if len(v) else np.nan,
        float(v.std(ddof=1)) if len(v) > 1 else np.nan,
    )


def _cell(m: float, s: float) -> str:
    if not np.isfinite(m):
        return "–"
    return f"{m:.3f}" + (f" ± {s:.3f}" if np.isfinite(s) else "")


def _is_primary(long: pd.DataFrame) -> pd.Series:
    want = [primary_score(m, b) for m, b in zip(long["method"], long["background"], strict=True)]
    return long["score"] == pd.Series(want, index=long.index)


def headline(long: pd.DataFrame, unknown_set: str) -> pd.DataFrame:
    """Mean and sd over runs per protocol and method for the HEADLINE metrics at the primary
    score."""
    long = long[_is_primary(long)]
    out = []
    for protocol in _order(long["protocol"].unique(), PROTOCOL_ORDER):
        lp = long[long["protocol"] == protocol]
        for method in _order(lp["method"].unique(), METHOD_ORDER):
            d = lp[lp["method"] == method]
            row: dict = {"protocol": protocol, "method": method, "score": d["score"].iloc[0]}
            row["n_runs"] = int(d["run_id"].nunique())
            for col, where, key in HEADLINE:
                sub = d[d["set"] == ("test_known" if where == "known" else unknown_set)]
                row[f"{col}_mean"], row[f"{col}_sd"] = _mean_sd(sub[key] if key in sub else [])
            out.append(row)
    return pd.DataFrame(out)


def by_score(long: pd.DataFrame) -> pd.DataFrame:
    """Per protocol, method and score: known-set numbers and, for each unknown set, the share
    accepted at tau and the AUROC (mean and sd over runs)."""
    out = []
    long = long.assign(primary=_is_primary(long))
    for protocol in _order(long["protocol"].unique(), PROTOCOL_ORDER):
        lp = long[long["protocol"] == protocol]
        for method in _order(lp["method"].unique(), METHOD_ORDER):
            lm = lp[lp["method"] == method]
            scores = sorted(
                lm["score"].unique(), key=lambda x: (not lm[lm["score"] == x]["primary"].any(), x)
            )
            for score in scores:
                d = lm[lm["score"] == score]
                known = d[d["set"] == "test_known"]
                row: dict = {
                    "protocol": protocol,
                    "method": method,
                    "score": score,
                    "primary": bool(d["primary"].any()),
                    "n_runs": int(d["run_id"].nunique()),
                }
                for col, key in (
                    ("accuracy", "accuracy"),
                    ("macro_f1", "macro_f1"),
                    ("ece", "ece"),
                    ("known_kept", "known_accept_rate"),
                ):
                    row[f"{col}_mean"], row[f"{col}_sd"] = _mean_sd(
                        known[key] if key in known else []
                    )
                for set_name in UNKNOWN_SETS:
                    sub = d[d["set"] == set_name]
                    short = set_name.removeprefix("test_")
                    for key in ("accepted", "auroc", "fpr95"):
                        row[f"{short}_{key}_mean"], row[f"{short}_{key}_sd"] = _mean_sd(
                            sub[key] if key in sub else []
                        )
                out.append(row)
    return pd.DataFrame(out)


def by_score_markdown(t: pd.DataFrame) -> str:
    cols = [
        ("Protocol", None), ("Method", None), ("Score", None), ("Runs", None),
        ("Known accuracy", "accuracy"), ("Known kept", "known_kept"),
        ("Unknown accepted ↓", "unknown_accepted"), ("Unknown FPR@95 ↓", "unknown_fpr95"),
        ("Unknown AUROC ↑", "unknown_auroc"), ("Non-exercise accepted ↓", "nonex_accepted"),
        ("Background accepted ↓", "background_accepted"),
        ("Look-alike accepted", "lookalike_accepted"), ("ECE ↓", "ece"),
    ]  # fmt: skip
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = [r["protocol"], r["method"], r["score"] + (" *" if r["primary"] else "")]
        cells.append(str(r["n_runs"]))
        cells += [_cell(r[f"{k}_mean"], r[f"{k}_sd"]) for _, k in cols[4:]]
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        "\nMean ± sd over runs (folds x class splits x seeds). `*` marks the method's primary "
        "score. Accepted = share of that set's test windows with score <= tau, where tau keeps "
        "95 % of the known validation windows; known kept = the same share for the known test "
        "windows; FPR@95 = the accepted share when the threshold instead keeps 95 % of the "
        "known test windows. Unknown = classes no model saw; non-exercise and background are "
        "categories "
        "that taught and Other-class models saw in training, from other subjects; look-alike "
        "= variants of a known exercise.\n"
    )
    return "\n".join(lines) + "\n" + note


def to_markdown(t: pd.DataFrame, unknown_set: str) -> str:
    """Markdown table with ``mean ± sd`` cells (sd omitted for a single run)."""
    names = {"macro_f1": "macro-F1 ↑", "auroc": "AUROC ↑", "fpr95": "FPR@95 ↓", "ece": "ECE ↓"}
    names |= {"aurc": "AURC ↓", "accepted": "Accepted at τ ↓"}
    cols = ["Protocol", "Method", "Score", "Runs", *(names[c] for c, _, _ in HEADLINE)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = [r["protocol"], r["method"], str(r["score"]), str(r["n_runs"])]
        cells += [_cell(r[f"{c}_mean"], r[f"{c}_sd"]) for c, _, _ in HEADLINE]
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        f"\nMean ± sd over the runs of one protocol. AUROC, FPR@95 and the share accepted at τ on "
        f"`{unknown_set}` against `test_known`; macro-F1, ECE and AURC on `test_known`. One "
        "primary score per method (`uaar.models.factory.primary_score`).\n"
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

    files = sorted(args.runs_dir.glob("*/metrics.json"))
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
    detail = by_score(long)
    detail.to_csv(args.out / "scores_by_protocol.csv", index=False)
    (args.out / "scores_by_protocol.md").write_text(by_score_markdown(detail))
    print(f"{len(files)} runs, {len(long)} rows -> {args.out}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
