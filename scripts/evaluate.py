#!/usr/bin/env python3
"""Evaluate one trained run on the P1 test sets (development-spec §3.5, §5).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

    uv run scripts/evaluate.py reports/runs/p1_recofit_edl_fold0_rand0_seed0 \
        --sets test_known test_unknown test_nonex

Loads the run with ``uaar.models.factory.load`` plus its own ``split.json``, ``config.yaml`` and
``threshold.json``, predicts on the requested sets with ``uaar.eval.runner.evaluate`` and writes
``preds_{set}.parquet`` and ``metrics.json`` into the run directory. The ``metrics.json`` layout
is documented in ``src/uaar/eval/runner.py``.

MC Dropout does not train, so it starts from a softmax run:

    uv run scripts/evaluate.py --method mcdropout \
        --from reports/runs/p1_recofit_softmax_fold0_rand0_seed0 [--T 30] [--out <run_dir>]

This first writes a self-contained run directory next to the softmax run (``_softmax_`` replaced
by ``_mcdropout_`` in the name unless ``--out`` is given): the same weights, split and
normalization, ``config.yaml`` with ``method: mcdropout`` and T, ``preds_val.parquet`` and
``threshold.json`` from T stochastic passes on the validation windows. It is then evaluated
like any other run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from uaar.config import Config
from uaar.data import registry, splits
from uaar.eval import runner
from uaar.eval.preds import write_preds
from uaar.models import factory
from uaar.models.mcdropout import MCDropoutMethod
from uaar.utils import git_sha, seed_everything


def _fmt(x) -> str:
    return "  nan" if x is None else f"{x:.3f}"


def make_mcdropout_run(source: Path, T: int | None, out: Path | None, chunk_size: int) -> Path:
    """Build the MC Dropout run directory from the softmax run ``source``; returns its path."""
    cfg = Config.from_yaml(source / "config.yaml")
    if cfg.method != "softmax":
        sys.exit(f"--from must be a softmax run, {source} is {cfg.method!r}")
    T = int(T if T is not None else cfg.get("model.mcdropout.T", 30))
    run_dir = out or source.parent / source.name.replace("_softmax_", "_mcdropout_")
    if run_dir.resolve() == source.resolve():
        sys.exit("the MC Dropout run directory would overwrite the softmax run; pass --out")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    method = MCDropoutMethod.from_softmax_run(source, T=T, seed=seed)
    method.save(run_dir)
    shutil.copy(source / "split.json", run_dir / "split.json")
    cfg.set("method", "mcdropout")
    cfg.set("model.mcdropout.T", T)
    cfg.set("run_id", run_dir.name)
    cfg.set("source_run", source.name)
    cfg.to_yaml(run_dir / "config.yaml")

    split = splits.load_split(run_dir / "split.json")
    X, meta = registry.load_windows(cfg.dataset)
    frame, taus = runner.validation_thresholds(method, X, meta, split, cfg, chunk_size)
    write_preds(frame, run_dir, "val")
    (run_dir / "threshold.json").write_text(
        json.dumps({"accept_rate": cfg.eval.accept_rate, "tau": taus}, indent=2)
    )
    print(f"{run_dir.name}: built from {source.name} with T={T}; tau {taus}")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, nargs="?")
    ap.add_argument("--method", choices=["mcdropout"], help="a method that does not train")
    ap.add_argument("--from", dest="source", type=Path, help="softmax run to take weights from")
    ap.add_argument("--T", type=int, default=None, help="stochastic passes (default 30)")
    ap.add_argument("--out", type=Path, default=None, help="run directory to create")
    ap.add_argument(
        "--sets",
        nargs="+",
        default=list(runner.SUPPORTED_SETS),
        choices=runner.SUPPORTED_SETS,
        help="test sets to evaluate (test_known is added when an unknown set is requested)",
    )
    ap.add_argument("--chunk-size", type=int, default=8192, help="windows per predict call")
    args = ap.parse_args()

    if args.method == "mcdropout":
        if args.source is None or args.run_dir is not None:
            ap.error("--method mcdropout needs --from <softmax run> and no positional run_dir")
        run_dir = make_mcdropout_run(args.source, args.T, args.out, args.chunk_size)
    elif args.run_dir is None or args.source is not None:
        ap.error("give a run directory, or --method mcdropout --from <softmax run>")
    else:
        run_dir = args.run_dir
    cfg = Config.from_yaml(run_dir / "config.yaml")
    split = splits.load_split(run_dir / "split.json")
    taus = json.loads((run_dir / "threshold.json").read_text())["tau"]
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    t0 = time.perf_counter()
    method = factory.load(run_dir)
    X, meta = registry.load_windows(cfg.dataset)
    result, frames = runner.evaluate(
        method, taus, X, meta, split, args.sets, cfg, chunk_size=args.chunk_size
    )
    for name, df in frames.items():
        write_preds(df, run_dir, name)
    elapsed = time.perf_counter() - t0

    header = {
        "run_id": cfg.get("run_id", run_dir.name),
        "method": result.pop("method"),
        "dataset": result.pop("dataset"),
        "protocol": cfg.get("protocol"),
        "split_id": result.pop("split_id"),
        "seed": seed,
        "K": result.pop("K"),
        "git_sha": cfg.get("git_sha", "unknown"),
        "eval_git_sha": git_sha(),
        "eval_seconds": round(elapsed, 2),
    }
    out = runner.jsonable(header | result)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    known = out.get(runner.KNOWN_SET, {})
    cs, cal = known.get("closed_set", {}), known.get("calibration", {})
    print(
        f"{header['run_id']}: {', '.join(f'{k} {len(v)}' for k, v in frames.items())} windows; "
        f"acc {_fmt(cs.get('accuracy'))} macro-F1 {_fmt(cs.get('macro_f1'))} "
        f"ECE {_fmt(cal.get('ece'))}; {elapsed:.1f} s"
    )
    for name in runner.UNKNOWN_SETS:
        for score, m in out.get(name, {}).get("scores", {}).items():
            print(
                f"  {name:<12} {score:<14} AUROC {_fmt(m['auroc'])} "
                f"FPR@95 {_fmt(m['fpr95'])} open-F1 {_fmt(m['open_macro_f1'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
