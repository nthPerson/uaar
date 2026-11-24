#!/usr/bin/env python3
"""Train and evaluate every run of a sweep file, one after the other (development-spec §5).

    uv run scripts/sweep.py configs/sweeps/p1_fold0.yaml [--dry-run] [--set key=value ...]

A sweep file lists ``folds``, ``class_splits``, ``seeds`` and ``methods``; the method configs
are ``configs/<protocol>/<method>.yaml`` (``--protocol``, default ``p1``). A file may instead
give ``protocols: {<protocol>: [methods], ...}`` to run several protocols (for example the
three tiers ``k4plain``, ``k4taught``, ``k4other``) on the same folds and class splits. For
each fold, class split and seed the methods run in the order listed: a trained method is
``train.py`` then ``evaluate.py``; ``mcdropout`` is ``evaluate.py --method mcdropout --from``
the softmax run of the same cell and protocol, so list ``softmax`` before it. A run whose
``metrics.json`` exists is skipped, so an interrupted sweep can be restarted with the same
command. A failed step stops that run, is reported at the end, and the sweep goes on.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import yaml

from uaar.config import Config

SCRIPTS = Path(__file__).resolve().parent


def run_id(cfg: Config, method: str, fold: int, class_split: str, seed: int) -> str:
    return f"{cfg.protocol}_{cfg.dataset}_{method}_fold{fold}_{class_split}_seed{seed}"


def steps_for(
    method: str, config: Path, fold: int, class_split: str, seed: int, runs_dir: Path,
    overrides: list[str], sets: list[str] | None,
) -> list[list[str]]:  # fmt: skip
    """The commands that produce one run, in order. ``sets=None`` leaves the choice of test
    sets to ``evaluate.py`` (every set the split has)."""
    cfg = Config.load(str(config), overrides)
    this = runs_dir / run_id(cfg, method, fold, class_split, seed)
    py = [sys.executable]
    evaluate = [*py, str(SCRIPTS / "evaluate.py")]
    sets = ["--sets", *sets] if sets else []
    if method == "mcdropout":
        source = runs_dir / run_id(cfg, "softmax", fold, class_split, seed)
        T = cfg.get("model.mcdropout.T", 30)
        return [
            [*evaluate, "--method", "mcdropout", "--from", str(source), "--T", str(T),
             "--out", str(this), *sets]
        ]  # fmt: skip
    train = [
        *py, str(SCRIPTS / "train.py"), "--config", str(config), "--fold", str(fold),
        "--class-split", class_split, "--seed", str(seed), "--runs-dir", str(runs_dir),
    ]  # fmt: skip
    if overrides:
        train += ["--set", *overrides]
    return [train, [*evaluate, str(this), *sets]]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--protocol", default="p1")
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--sets", nargs="+", default=None, help="default: every set of the split")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    sweep = yaml.safe_load(args.sweep.read_text())
    cells = list(itertools.product(sweep["folds"], sweep["class_splits"], sweep["seeds"]))
    protocols = sweep.get("protocols") or {args.protocol: sweep["methods"]}
    for methods in protocols.values():
        if "mcdropout" in methods and "softmax" not in methods[: methods.index("mcdropout")]:
            sys.exit("list softmax before mcdropout: MC Dropout uses the softmax run of the cell")
    plan = [(p, m) for p, methods in protocols.items() for m in methods]

    done, skipped, failed = [], [], []
    t_all = time.perf_counter()
    for fold, class_split, seed in cells:
        for protocol, method in plan:
            config = args.configs_dir / protocol / f"{method}.yaml"
            cfg = Config.load(str(config), args.overrides)
            rid = run_id(cfg, method, fold, class_split, seed)
            if (args.runs_dir / rid / "metrics.json").exists():
                skipped.append(rid)
                print(f"-- {rid}: done already, skipped", flush=True)
                continue
            steps = steps_for(
                method, config, fold, class_split, seed, args.runs_dir, args.overrides, args.sets
            )
            t0 = time.perf_counter()
            ok = True
            for cmd in steps:
                print("$ " + " ".join(cmd[1:]), flush=True)
                if args.dry_run:
                    continue
                if subprocess.run(cmd).returncode != 0:
                    ok = False
                    break
            if args.dry_run:
                continue
            (done if ok else failed).append(rid)
            state = "ok" if ok else "FAILED"
            print(f"-- {rid}: {state} in {time.perf_counter() - t0:.0f} s", flush=True)
    print(
        f"sweep: {len(done)} run, {len(skipped)} skipped, {len(failed)} failed "
        f"in {(time.perf_counter() - t_all) / 60:.1f} min"
    )
    for rid in failed:
        print(f"  failed: {rid}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
