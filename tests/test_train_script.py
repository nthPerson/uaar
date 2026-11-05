"""train.py end to end on a tiny synthetic processed dataset (spec §7 test_train_smoke)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from uaar.data import splits


def _make_dataset(root: Path, n_subjects=12, n_classes=4, per=12):
    rng = np.random.default_rng(0)
    t = np.arange(100) / 50
    X, rows = [], []
    for s in range(n_subjects):
        for k in range(n_classes):
            for i in range(per):
                sig = np.sin(2 * np.pi * (0.5 + k) * t + rng.uniform(0, 6.28))
                X.append(
                    np.stack(
                        [
                            sig,
                            0.3 * sig,
                            np.ones(100),
                            50 * np.cos(2 * np.pi * (0.5 + k) * t),
                            np.zeros(100),
                            5 * sig,
                        ]
                    )
                    + rng.normal(0, 0.05, (6, 100))
                )
                rows.append(
                    {
                        "subject": f"S{s:02d}",
                        "recording_id": f"single/S{s:02d}/{k}/{i}",
                        "source": "single",
                        "label": f"class{k}",
                        "start_s": float(i),
                        "label_purity": 1.0,
                        "n_gaps": 0,
                    }
                )
    meta = pd.DataFrame(rows)
    meta.insert(0, "window_id", np.arange(len(meta)))
    d = root / "processed" / "recofit" / "windows"
    d.mkdir(parents=True)
    np.save(d / "X.npy", np.asarray(X, np.float32))
    meta.to_parquet(d / "meta.parquet", index=False)
    splits.write_splits(
        meta,
        root / "processed" / "recofit" / "splits",
        n_folds=3,
        class_split_seeds=(0,),
        min_subjects=2,
        min_minutes=0.1,
        dataset="recofit",
    )


def test_train_script_writes_run_dir(tmp_path):
    _make_dataset(tmp_path / "data")
    env = {**os.environ, "UAAR_DATA_DIR": str(tmp_path / "data")}
    for method in ("softmax", "edl"):
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            f"configs/p1/{method}.yaml",
            "--fold",
            "0",
            "--class-split",
            "rand0",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--set",
            "train.epochs=3",
            "train.batch_size=32",
            "train.patience=2",
        ]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        run_dir = tmp_path / "runs" / f"p1_recofit_{method}_fold0_rand0_seed0"
        for f in (
            "config.yaml",
            "split.json",
            "model.pt",
            "config.json",
            "preprocess.json",
            "history.csv",
            "preds_val.parquet",
            "threshold.json",
        ):
            assert (run_dir / f).exists(), f
        preds = pq.read_table(run_dir / "preds_val.parquet").to_pandas()
        assert {"window_id", "y_true", "y_pred", "p_max", "probs"} <= set(preds.columns)
        tau = json.loads((run_dir / "threshold.json").read_text())["tau"]
        assert set(tau) == (
            {"msp", "entropy"} if method == "softmax" else {"vacuity", "one_minus_max"}
        )


def test_train_and_evaluate_logreg(tmp_path):
    """logreg goes through the same scripts: grouped C selection, reload, evaluation."""
    _make_dataset(tmp_path / "data")
    env = {**os.environ, "UAAR_DATA_DIR": str(tmp_path / "data")}
    cmd = [
        sys.executable, "scripts/train.py", "--config", "configs/p1/logreg.yaml",
        "--fold", "0", "--class-split", "rand0", "--runs-dir", str(tmp_path / "runs"),
        "--set", "model.logreg.C_grid=[0.1,1.0]", "model.logreg.n_jobs=1",
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    run_dir = tmp_path / "runs" / "p1_recofit_logreg_fold0_rand0_seed0"
    for f in ("config.yaml", "split.json", "model.joblib", "config.json", "preprocess.json",
              "history.csv", "preds_val.parquet", "threshold.json"):  # fmt: skip
        assert (run_dir / f).exists(), f
    hist = pd.read_csv(run_dir / "history.csv")
    assert set(hist["mode"]) == {"group_kfold3"}  # train.py passed the training subjects
    assert list(hist["C"]) == [0.1, 1.0] and hist["chosen"].sum() == 1
    preds = pq.read_table(run_dir / "preds_val.parquet").to_pandas()
    assert (preds["y_pred"] == preds["y_true"]).mean() > 0.9  # classes differ in frequency

    ev = [sys.executable, "scripts/evaluate.py", str(run_dir), "--sets", "test_known",
          "test_unknown"]  # fmt: skip
    out = subprocess.run(ev, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["method"] == "logreg" and set(m["score_names"]) == {"msp", "entropy"}
    assert m["test_known"]["closed_set"]["accuracy"] > 0.9
