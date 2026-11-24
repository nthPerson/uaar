#!/usr/bin/env python3
"""Fine-tune a RecoFit-trained model on our own wearable recordings and report before/after.

    uv run scripts/finetune.py --run models/k4taught_recofit_edl_fold0_bg0_seed0 \
        --train data/wearable/W01 --test data/wearable/W02

``--run`` is a trained softmax or EDL run directory. ``--train`` and ``--test`` are subject
folders (every session under them is used); swap them for the other direction of the
leave-one-person-out test. Without ``--test`` the model is only fine-tuned and saved.

What it does (docs/decisions.md, 2026-11-23): the convolutional trunk is frozen, the input
normalization is re-fitted on the training person's windows, the two output layers are trained
on their known-exercise sets, and a taught model also keeps being told to be uncertain on
their background movements. Validation is the last set of every label; it gives early
stopping and the new rejection thresholds.

Output, in ``models/`` (tracked in git), a run directory the live page can load
(``configs/demo.yaml``):

    <out>/model.pt, config.json, preprocess.json   the fine-tuned model
    <out>/threshold.json                           thresholds from the wearable validation sets
    <out>/history.csv, finetune.json               training curve, settings, window counts
    <out>/report.json, report.md                   before/after on the test person

For a softmax ``--run`` the MC Dropout model built on the fine-tuned network is saved next to
it (``..._mcdropout_...``) and reported as well. Labels and their roles: ``configs/wearable.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from uaar.data.wearable_windows import Roles, subject_windows, validation_split
from uaar.eval import threshold
from uaar.models import factory
from uaar.models.finetune import finetune
from uaar.models.mcdropout import MCDropoutMethod

ROOT = Path(__file__).resolve().parents[1]


def encode(table: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Class index of every window (-1 where the window has no model class)."""
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index.get(c, -1) for c in table["class_name"]], dtype=np.int64)


def check_classes(table: pd.DataFrame, classes: list[str], who: str) -> None:
    named = set(table.loc[table["role"].isin(["known", "shifted"]), "class_name"])
    missing = named - set(classes)
    if missing:
        sys.exit(f"{who}: labels map to classes the model does not have: {sorted(missing)}")


def thresholds(method, X: np.ndarray, accept: float) -> dict[str, float]:
    """The ``accept`` quantile of every score on known validation windows."""
    scores = method.predict(X).scores
    return {name: threshold.select_tau(s, accept) for name, s in scores.items()}


def evaluate(method, taus: dict[str, float], score: str, X: np.ndarray, table: pd.DataFrame):
    """Metrics of one model on one person's windows, for its primary score.

    Returns a dict: ``accuracy`` and ``macro_f1`` on the known sets, ``known_kept`` (share of
    known windows accepted), and the share accepted of the unknown movements (never trained
    on), the background movements (trained on for the other person, if taught) and the
    known exercise recorded with the kit worn differently (with its accuracy).
    """
    y = encode(table, list(method.classes))
    pred = method.predict(X)
    accepted = pred.scores[score] <= taus[score]
    out: dict[str, float | int] = {}
    known = (table["role"] == "known").to_numpy()
    if known.any():
        out["n_known"] = int(known.sum())
        out["accuracy"] = float((pred.label[known] == y[known]).mean())
        labels = sorted(set(y[known]))
        out["macro_f1"] = float(
            f1_score(y[known], pred.label[known], average="macro", labels=labels)
        )
        out["known_kept"] = float(accepted[known].mean())
    for role, key in (("unknown", "unknown_accepted"), ("background", "background_accepted")):
        rows = (table["role"] == role).to_numpy()
        if rows.any():
            out[f"n_{role}"] = int(rows.sum())
            out[key] = float(accepted[rows].mean())
    shifted = (table["role"] == "shifted").to_numpy()
    if shifted.any():
        out["n_shifted"] = int(shifted.sum())
        out["shifted_accuracy"] = float((pred.label[shifted] == y[shifted]).mean())
        out["shifted_kept"] = float(accepted[shifted].mean())
    return out


COLUMNS = [
    ("accuracy", "known accuracy"),
    ("macro_f1", "macro-F1"),
    ("known_kept", "known kept"),
    ("unknown_accepted", "unknown movements accepted"),
    ("background_accepted", "rest and taught movements accepted"),
    ("shifted_accuracy", "shifted-kit accuracy"),
    ("shifted_kept", "shifted-kit kept"),
]


def to_markdown(rows: list[dict]) -> str:
    cols = [(k, title) for k, title in COLUMNS if any(k in r for r in rows)]
    lines = [
        "| model | score | " + " | ".join(t for _, t in cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for r in rows:
        cells = [f"{r[k]:.3f}" if k in r else "" for k, _ in cols]
        lines.append(f"| {r['model']} | {r['score']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_run(method, run_dir: Path, taus: dict, accept: float, info: dict, history=None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    method.save(run_dir)
    (run_dir / "threshold.json").write_text(
        json.dumps({"accept_rate": accept, "tau": taus, "source": "wearable validation"}, indent=2)
    )
    (run_dir / "finetune.json").write_text(json.dumps(info, indent=2))
    if history is not None:
        history.write(run_dir / "history.csv")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True, help="trained softmax or EDL run")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="subject folder(s)")
    ap.add_argument("--test", type=Path, nargs="*", default=[], help="subject folder(s)")
    ap.add_argument("--roles", type=Path, default=ROOT / "configs" / "wearable.yaml")
    ap.add_argument("--out", type=Path, default=None, help="default: models/<run>_ft-<subjects>")
    ap.add_argument("--full", action="store_true", help="also train the convolutional trunk")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--accept", type=float, default=0.95, help="share of known windows kept")
    ap.add_argument("--T", type=int, default=30, help="MC Dropout passes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    roles = Roles.load(args.roles)
    source = factory.load(args.run)
    if source.name not in ("softmax", "edl"):
        sys.exit(f"--run must be a softmax or EDL run, got {source.name}")
    background = getattr(source, "background", "none")
    classes = list(source.classes)
    tag = "+".join(p.name for p in args.train)
    out = args.out or ROOT / "models" / f"{args.run.name}_ft-{tag}"

    Xw, tw = subject_windows(args.train, roles)
    check_classes(tw, classes, "--train")
    if tw.attrs["ignored"]:
        print(f"ignored labels without a role: {tw.attrs['ignored']}")
    train, val = validation_split(tw)
    y = encode(tw, classes)
    known = (tw["role"] == "known").to_numpy()
    bg = (tw["role"] == "background").to_numpy()
    counts = {
        "train_known": int((known & train).sum()),
        "val_known": int((known & val).sum()),
        "train_background": int((bg & train).sum()),
        "per_class_train": {
            c: int(((y == i) & known & train).sum()) for i, c in enumerate(classes)
        },
    }
    print(f"{args.run.name}: {source.name}, background mode {background}")
    print(f"train on {tag}: {counts}")
    empty = [c for c, n in counts["per_class_train"].items() if n == 0]
    if empty:
        sys.exit(f"no training windows for: {empty} (check the labels against {args.roles})")

    # ---- before: the RecoFit-trained model with its RecoFit thresholds, on the test person
    rows, score = [], factory.primary_score(source.name, background)
    src_taus = json.loads((args.run / "threshold.json").read_text())["tau"]
    Xt = tt = None
    if args.test:
        Xt, tt = subject_windows(args.test, roles)
        check_classes(tt, classes, "--test")
        before = evaluate(source, src_taus, score, Xt, tt)
        rows.append({"model": f"{source.name}, RecoFit only", "score": score, **before})

    # ---- fine-tune (the loaded model is changed in place)
    history = finetune(
        source,
        Xw[known & train],
        y[known & train],
        Xw[known & val],
        y[known & val],
        Xw[bg & train],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        freeze=not args.full,
        seed=args.seed,
    )
    taus = thresholds(source, Xw[known & val], args.accept)
    info = {
        "source_run": args.run.name,
        "train_subjects": [p.name for p in args.train],
        "test_subjects": [p.name for p in args.test],
        "recipe": "normalization re-fitted; "
        + ("all layers trained" if args.full else "trunk frozen, fc and head trained"),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "best_epoch": history.best_epoch,
        "best_val_macro_f1": history.best_val_f1,
        "counts": counts,
    }
    save_run(source, out, taus, args.accept, info, history)
    print(f"fine-tuned: best epoch {history.best_epoch}, val macro-F1 {history.best_val_f1:.3f}")
    print(f"saved {out}")
    if Xt is not None:
        rows.append(
            {
                "model": f"{source.name}, fine-tuned",
                "score": score,
                **evaluate(source, taus, score, Xt, tt),
            }
        )

    # ---- MC Dropout on the fine-tuned softmax network
    if source.name == "softmax":
        mc = MCDropoutMethod.from_softmax(source, T=args.T, seed=args.seed, source_run=out.name)
        mc_score = factory.primary_score("mcdropout", background)
        mc_taus = thresholds(mc, Xw[known & val], args.accept)
        mc_out = out.with_name(
            out.name.replace("_softmax_", "_mcdropout_")
            if "_softmax_" in out.name
            else out.name + "-mcdropout"
        )
        save_run(mc, mc_out, mc_taus, args.accept, {**info, "T": args.T})
        print(f"saved {mc_out}")
        if Xt is not None:
            rows.append(
                {
                    "model": "mcdropout, fine-tuned",
                    "score": mc_score,
                    **evaluate(mc, mc_taus, mc_score, Xt, tt),
                }
            )

    if rows:
        md = to_markdown(rows)
        head = f"Trained on {tag}, tested on {'+'.join(p.name for p in args.test)}.\n\n"
        (out / "report.md").write_text(head + md + "\n")
        (out / "report.json").write_text(json.dumps(rows, indent=2))
        print("\n" + head + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
