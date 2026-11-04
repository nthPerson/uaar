#!/usr/bin/env python3
"""download -> recordings -> windows -> splits (-> features) for one dataset; idempotent.

    uv run scripts/prepare_data.py recofit --steps all
    uv run scripts/prepare_data.py recofit --steps recordings windows --force

Each step skips when its outputs exist unless ``--force``. ``labels`` prints the per-label
minutes and subject counts from ``recordings.meta.json`` (the T1 input and the list the
known/unknown and ``near`` decisions are made from).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from uaar.config import Config
from uaar.data import registry, splits, windows
from uaar.utils import timer

STEPS = ["download", "recordings", "windows", "splits", "features"]


def step_download(dataset: str, p: dict, force: bool) -> None:
    if dataset == "recofit":
        from uaar.data import recofit

        recofit.download(p["external"], force=force)
    else:
        print(f"{dataset}: download not implemented yet; place files under {p['external']}")


def step_recordings(dataset: str, p: dict, force: bool, limit: int | None) -> None:
    if p["recordings"].exists() and not force:
        print(f"skip recordings ({p['recordings']} exists)")
        return
    if dataset == "recofit":
        from uaar.data import recofit

        single = p["external"] / "exercise_data.50.0000_singleonly.mat"
        multi = p["external"] / "exercise_data.50.0000_multionly.mat"
        for f in (single, multi):
            if not f.exists():
                sys.exit(f"missing {f}; run --steps download")
        with timer("recordings"):
            recofit.to_recordings(p["root"], single, multi, hash_sources=limit is None)
    elif dataset == "wearable":
        from uaar.data import wearable

        with timer("recordings"):
            wearable.to_recordings(registry.data_root() / "wearable", p["root"])
    else:
        sys.exit(f"{dataset}: recordings step not implemented yet")


def step_windows(p: dict, cfg: Config, force: bool) -> None:
    if p["windows_x"].exists() and not force:
        print(f"skip windows ({p['windows_x']} exists)")
        return
    with timer("windows"):
        windows.make_windows(
            p["recordings"],
            p["windows_dir"],
            window_s=cfg.data.window_s,
            hop_s=cfg.data.hop_s,
            purity=cfg.data.purity,
        )


def step_splits(dataset: str, p: dict, cfg: Config, force: bool, near_file: Path | None) -> None:
    if p["splits_dir"].exists() and any(p["splits_dir"].glob("*.json")) and not force:
        print(f"skip splits ({p['splits_dir']} has files)")
        return
    _, meta = registry.load_windows(dataset)
    near_pairs = None
    if near_file is not None and near_file.exists():
        near_pairs = yaml.safe_load(near_file.read_text()).get("pairs")
    with timer("splits"):
        written = splits.write_splits(
            meta,
            p["splits_dir"],
            n_folds=cfg.split.n_folds,
            near_pairs=near_pairs,
            min_subjects=cfg.split.min_subjects,
            min_minutes=cfg.split.min_minutes,
            frac_unknown=cfg.split.frac_unknown,
            val_frac=cfg.split.val_frac,
            dataset=dataset,
            hop_s=cfg.data.hop_s,
            exclude=cfg.get("split.nonexercise_labels", []) or [],
        )
    first = json.loads(written[0].read_text())
    n_eligible = len(first["known_classes"]) + len(first["unknown_classes"])
    print(f"{len(written)} split files; eligible classes: {n_eligible}")
    print("known  :", first["known_classes"])
    print("unknown:", first["unknown_classes"])


def step_features(dataset: str, p: dict, force: bool) -> None:
    try:
        from uaar.features import handcrafted
    except ImportError:
        print("features: uaar.features.handcrafted not written yet (Sirena, item 5.1); skipped")
        return
    if p["features"].exists() and not force:
        print(f"skip features ({p['features']} exists)")
        return
    X, meta = registry.load_windows(dataset)
    with timer("features"):
        df = handcrafted.compute(X)
    df.insert(0, "window_id", meta["window_id"].to_numpy())
    p["features_dir"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(p["features"], index=False)
    print(f"features: {df.shape}")


def print_labels(p: dict) -> None:
    from uaar.data.recofit import print_label_table

    print_label_table(p["recordings_meta"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", choices=["recofit", "mmfit", "recgym", "wearable"])
    ap.add_argument("--steps", nargs="+", default=["all"], choices=STEPS + ["all", "labels"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--config", default=None, help="method/protocol yaml (defaults to base.yaml only)"
    )
    ap.add_argument(
        "--set", dest="overrides", nargs="*", default=[], help="key.path=value overrides"
    )
    ap.add_argument("--near", type=Path, default=Path("configs/near_unknown_recofit.yaml"))
    ap.add_argument(
        "--limit", type=int, default=None, help="(debug) subjects limit, unused for now"
    )
    args = ap.parse_args()

    cfg = Config.load(args.config, args.overrides)
    p = registry.paths(args.dataset)
    steps = STEPS if "all" in args.steps else args.steps
    if "labels" in args.steps:
        print_labels(p)
        return
    for s in steps:
        print(f"== {s}")
        if s == "download":
            step_download(args.dataset, p, args.force)
        elif s == "recordings":
            step_recordings(args.dataset, p, args.force, args.limit)
        elif s == "windows":
            step_windows(p, cfg, args.force)
        elif s == "splits":
            step_splits(args.dataset, p, cfg, args.force, args.near)
        elif s == "features":
            step_features(args.dataset, p, args.force)


if __name__ == "__main__":
    main()
