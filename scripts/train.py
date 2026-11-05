#!/usr/bin/env python3
"""Train one run: config -> split -> fit -> run directory (development-spec §3.5, §5).

    uv run scripts/train.py --config configs/p1/edl.yaml --fold 0 --class-split rand0 --seed 0

Writes reports/runs/<run_id>/ with config.yaml, split.json, model + config.json +
preprocess.json, history.csv, preds_val.parquet and threshold.json.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import time
from pathlib import Path

from uaar.config import Config
from uaar.data import registry, splits
from uaar.eval import threshold
from uaar.eval.preds import preds_frame
from uaar.eval.runner import window_positions
from uaar.models import factory
from uaar.utils import git_sha, seed_everything


def run_id_for(cfg: Config, split_id: str, seed: int) -> str:
    return f"{cfg.protocol}_{cfg.dataset}_{cfg.method}_{split_id}_seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--class-split", default="rand0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    ap.add_argument("--force", action="store_true", help="retrain even if the run exists")
    args = ap.parse_args()

    cfg = Config.load(args.config, args.overrides)
    split_id = f"fold{args.fold}_{args.class_split}"
    run_id = run_id_for(cfg, split_id, args.seed)
    run_dir = args.runs_dir / run_id
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"{run_id}: exists, skipping (use --force)")
        return 0
    if cfg.method == "mcdropout":
        sys.exit(
            "mcdropout does not train; use evaluate.py --method mcdropout --from <softmax run>"
        )

    seed_everything(args.seed)
    split_path = registry.split_path(cfg.dataset, split_id)
    split = splits.load_split(split_path)
    X, meta = registry.load_windows(cfg.dataset)
    # select() returns window ids; X and meta are indexed by row position
    tr_ids, va_ids = splits.select(meta, split, "train"), splits.select(meta, split, "val")
    tr, va = window_positions(meta, tr_ids), window_positions(meta, va_ids)
    labels = meta["label"].to_numpy()
    ytr = splits.encode_labels(labels[tr], split)
    yva = splits.encode_labels(labels[va], split)
    assert (ytr >= 0).all() and (yva >= 0).all(), "unknown-class window leaked into train/val"
    print(f"{run_id}: train {len(tr)} windows, val {len(va)}, K={len(split.known_classes)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.set("run_id", run_id)
    cfg.set("git_sha", git_sha())
    cfg.set("seed", args.seed)
    cfg.to_yaml(run_dir / "config.yaml")
    shutil.copy(split_path, run_dir / "split.json")

    method = factory.build(cfg, list(split.known_classes), seed=args.seed)
    t0 = time.perf_counter()
    fit_kwargs = {}
    if "groups" in inspect.signature(method.fit).parameters:
        # methods with an inner cross-validation (logreg) fold by training subject
        fit_kwargs["groups"] = meta["subject"].astype(str).to_numpy()[tr]
    method.fit(X[tr], ytr, X[va], yva, cfg, run_dir, **fit_kwargs)
    train_s = time.perf_counter() - t0
    method.save(run_dir)

    pred = method.predict(X[va], cfg)
    df = preds_frame(va_ids, yva, pred)
    df.to_parquet(run_dir / "preds_val.parquet", index=False)
    taus = {
        name: threshold.select_tau(pred.scores[name], cfg.eval.accept_rate)
        for name in method.score_names
    }
    (run_dir / "threshold.json").write_text(
        json.dumps({"accept_rate": cfg.eval.accept_rate, "tau": taus}, indent=2)
    )
    val_acc = float((pred.label == yva).mean())
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {"run_id": run_id, "train_seconds": train_s, "val_accuracy": val_acc, "tau": taus},
            indent=2,
        )
    )
    print(f"done in {train_s:.0f} s; val accuracy {val_acc:.3f}; tau {taus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
