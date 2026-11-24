#!/usr/bin/env python3
"""Copy the final version of a trained model into ``models/``, which is tracked in git.

    uv run scripts/export_model.py reports/runs/k4taught_recofit_edl_fold0_bg0_seed0
    uv run scripts/export_model.py <run_dir> [<run_dir> ...] [--models-dir models]

``reports/runs/`` holds every run of every sweep and is not in git. A model that the live page
or the report depends on is exported: its weights, the files needed to load it
(``config.json``, ``preprocess.json``, ``threshold.json``) and the small records of how it was
made (``config.yaml``, ``split.json``, ``train_summary.json``, ``metrics.json``,
``finetune.json``, ``report.md``, ``history.csv``). Predictions and plots stay behind. The
exported folder loads like a run directory (``uaar.models.factory.load``, ``configs/demo.yaml``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("config.json", "threshold.json")
WEIGHTS = ("model.pt", "model.joblib")
OPTIONAL = (
    "preprocess.json",
    "config.yaml",
    "split.json",
    "train_summary.json",
    "metrics.json",
    "finetune.json",
    "report.md",
    "report.json",
    "history.csv",
)


def export(run_dir: Path, models_dir: Path) -> Path:
    """Copy one run directory's model files to ``models_dir/<run name>``; returns that path."""
    run_dir = Path(run_dir)
    missing = [f for f in REQUIRED if not (run_dir / f).exists()]
    weights = [f for f in WEIGHTS if (run_dir / f).exists()]
    if missing or not weights:
        raise FileNotFoundError(
            f"{run_dir} is not a finished run: missing {missing or 'the weights file'}"
        )
    out = Path(models_dir) / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED, *weights, *OPTIONAL):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, out / name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", type=Path, nargs="+", help="run directories")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    for run in args.runs:
        try:
            out = export(run, args.models_dir)
        except FileNotFoundError as e:
            print(f"export: {e}", file=sys.stderr)
            return 1
        size = sum(p.stat().st_size for p in out.iterdir()) / 1024
        print(f"{out.relative_to(args.models_dir.parent)}  ({size:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
