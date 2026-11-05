"""evaluate.py and make_tables.py end to end on synthetic data; runner with a non-factory method."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uaar.config import Config
from uaar.data import splits
from uaar.eval import preds, runner
from uaar.models.base import Prediction

NONEX = ("Non-Exercise", "Rest")
SETS = ("test_known", "test_unknown", "test_nonex")


def _make_dataset(root: Path, n_subjects=12, n_classes=5, per=10, per_nonex=6):
    """Sine windows per class (single) plus noise-only Non-Exercise windows (multi)."""
    rng = np.random.default_rng(0)
    t = np.arange(100) / 50
    X, rows = [], []

    def add(sig, subject, label, source, i):
        X.append(
            np.stack([sig, 0.3 * sig, np.ones(100), 50 * sig, np.zeros(100), 5 * sig])
            + rng.normal(0, 0.05, (6, 100))
        )
        rows.append(
            {
                "subject": subject,
                "recording_id": f"{source}/{subject}/{label}/{i}",
                "source": source,
                "label": label,
                "start_s": float(i),
                "label_purity": 1.0,
                "n_gaps": 0,
            }
        )

    for s in range(n_subjects):
        subject = f"S{s:02d}"
        for k in range(n_classes):
            for i in range(per):
                add(
                    np.sin(2 * np.pi * (0.5 + k) * t + rng.uniform(0, 6.28)),
                    subject,
                    f"c{k}",
                    "single",
                    i,
                )
        for i in range(per_nonex):
            add(rng.normal(0, 0.5, 100), subject, "Non-Exercise", "multi", i)
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
        exclude=NONEX,
    )
    return meta


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    meta = _make_dataset(tmp / "data")
    env = {**os.environ, "UAAR_DATA_DIR": str(tmp / "data")}
    train = [
        sys.executable, "scripts/train.py", "--config", "configs/p1/softmax.yaml",
        "--fold", "0", "--class-split", "rand0", "--runs-dir", str(tmp / "runs"),
        "--set", "train.epochs=2", "train.batch_size=32", "train.patience=2",
    ]  # fmt: skip
    out = subprocess.run(train, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    run_dir = tmp / "runs" / "p1_recofit_softmax_fold0_rand0_seed0"
    ev = [sys.executable, "scripts/evaluate.py", str(run_dir), "--sets", *SETS, "--chunk-size", "7"]
    out = subprocess.run(ev, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return tmp, run_dir, meta, env


def test_evaluate_writes_preds_and_metrics(evaluated):
    _, run_dir, meta, _ = evaluated
    split = splits.load_split(run_dir / "split.json")
    val_cols = list(pd.read_parquet(run_dir / "preds_val.parquet").columns)
    for s in SETS:
        df = preds.read_preds(run_dir, s)
        assert list(df.columns) == val_cols  # same layout as train.py's preds_val
        assert len(df) > 0 and np.stack(df["probs"]).shape == (len(df), len(split.known_classes))
    known = preds.read_preds(run_dir, "test_known")
    assert (known["y_true"] >= 0).all()
    assert (preds.read_preds(run_dir, "test_unknown")["y_true"] == -1).all()
    nonex = preds.read_preds(run_dir, "test_nonex")
    expected = meta[(meta["label"] == "Non-Exercise") & meta["subject"].isin(split.test_subjects)]
    assert sorted(nonex["window_id"]) == sorted(expected["window_id"])
    assert preds.available_sets(run_dir) == sorted(["val", *SETS])

    m = json.loads((run_dir / "metrics.json").read_text())
    for key in ("run_id", "method", "split_id", "seed", "K", "git_sha", "eval_git_sha", "tau"):
        assert key in m, key
    assert m["method"] == "softmax" and m["split_id"] == "fold0_rand0" and m["seed"] == 0
    assert m["K"] == len(split.known_classes) and set(m["score_names"]) == {"msp", "entropy"}
    k = m["test_known"]
    assert set(k) == {"n", "closed_set", "calibration", "scores"} and k["n"] == len(known)
    assert set(k["closed_set"]) == {"accuracy", "macro_f1", "n"}
    assert set(k["calibration"]) == {"ece", "nll", "brier"}
    assert set(k["scores"]) == {"msp", "entropy"}
    assert set(k["scores"]["msp"]) == {
        "mean", "known_accept_rate", "misclassification_auroc", "aurc", "eaurc"
    }  # fmt: skip
    for s in ("test_unknown", "test_nonex"):
        row = m[s]["scores"]["msp"]
        assert set(row) == {
            "mean", "auroc", "aupr", "fpr95", "ratio", "n_known", "n_unknown",
            "open_macro_f1", "unknown_recall", "known_accept_rate",
        }  # fmt: skip
        assert 0.0 <= row["auroc"] <= 1.0 and row["n_known"] == len(known)
    assert 0.0 <= k["closed_set"]["accuracy"] <= 1.0


def test_metrics_recomputable_from_saved_preds(evaluated):
    _, run_dir, _, _ = evaluated
    m = json.loads((run_dir / "metrics.json").read_text())
    frames = {s: preds.read_preds(run_dir, s) for s in SETS}
    again = runner.jsonable(
        runner.compute_metrics(frames, m["tau"], m["score_names"], m["K"], m["accept_rate"])
    )
    for s in SETS:
        assert again[s].keys() == m[s].keys()
        assert again[s]["scores"]["msp"] == pytest.approx(m[s]["scores"]["msp"], nan_ok=True)


def test_make_tables(evaluated):
    tmp, _, _, env = evaluated
    out_dir = tmp / "tables"
    cmd = [
        sys.executable, "scripts/make_tables.py", "--runs-dir", str(tmp / "runs"),
        "--out", str(out_dir),
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    t2 = pd.read_csv(out_dir / "t2_headline.csv")
    assert t2["method"].tolist() == ["softmax"] and t2["score"].tolist() == ["msp"]
    assert {"macro_f1_mean", "auroc_mean", "fpr95_mean", "ece_mean", "aurc_mean"} <= set(t2)
    assert "| softmax | msp | 1 |" in (out_dir / "t2_headline.md").read_text()
    long = pd.read_csv(out_dir / "metrics_long.csv")
    assert len(long) == 3 * 2  # 1 run x 3 sets x 2 scores


class _MeanMethod:
    """A method that is not from factory.load: probs from per-channel means (no torch)."""

    name = "fake"
    score_names = ["u"]

    def __init__(self):
        self.calls = []

    def predict(self, X, cfg=None):
        self.calls.append(len(X))
        z = X.mean(axis=2)[:, :3]
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return Prediction(p, p.argmax(1), {"u": 1 - p.max(1), "extra": p.min(1)})


def test_runner_accepts_any_method_and_chunks(tmp_path):
    meta = _make_dataset(tmp_path / "data")
    X = np.load(tmp_path / "data/processed/recofit/windows/X.npy", mmap_mode="r")
    split = splits.load_split(tmp_path / "data/processed/recofit/splits/fold0_rand0.json")
    split.known_classes = split.known_classes[:3]  # 3 probs columns to match the fake method
    cfg = Config({"split": {"nonexercise_labels": list(NONEX)}, "eval": {"accept_rate": 0.95}})

    big = _MeanMethod()
    m1, f1 = runner.evaluate(big, {"u": 0.5}, X, meta, split, ["test_unknown"], cfg)
    small = _MeanMethod()
    m2, f2 = runner.evaluate(small, {"u": 0.5}, X, meta, split, ["test_unknown"], cfg, 5)
    assert list(f1) == ["test_known", "test_unknown"]  # reference set added automatically
    assert max(small.calls) <= 5 and len(small.calls) > len(big.calls)
    for s in f1:
        pd.testing.assert_frame_equal(f1[s].drop(columns="probs"), f2[s].drop(columns="probs"))
    assert m1 == m2 and m1["method"] == "fake" and set(m1["test_unknown"]["scores"]) == {"u"}

    with pytest.raises(KeyError, match="no tau"):
        runner.evaluate(_MeanMethod(), {}, X, meta, split, ["test_known"], cfg)
    with pytest.raises(ValueError, match="unsupported"):
        runner.evaluate(big, {"u": 0.5}, X, meta, split, ["shift_mmfit"], cfg)


def test_mcdropout_from_softmax_run(evaluated, tmp_path):
    _, softmax_run, _, env = evaluated
    out_dir = tmp_path / "p1_recofit_mcdropout_fold0_rand0_seed0"
    cmd = [
        sys.executable, "scripts/evaluate.py", "--method", "mcdropout", "--from", str(softmax_run),
        "--T", "5", "--out", str(out_dir), "--sets", *SETS, "--chunk-size", "50",
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for f in ("config.yaml", "config.json", "split.json", "model.pt", "preprocess.json",
              "threshold.json", "preds_val.parquet", "metrics.json"):  # fmt: skip
        assert (out_dir / f).exists(), f
    cfg = Config.from_yaml(out_dir / "config.yaml")
    assert cfg.method == "mcdropout" and cfg.get("model.mcdropout.T") == 5
    assert cfg.get("source_run") == softmax_run.name and cfg.get("run_id") == out_dir.name
    scores = {"entropy", "mi", "varratio"}
    tau = json.loads((out_dir / "threshold.json").read_text())["tau"]
    assert set(tau) == scores
    m = json.loads((out_dir / "metrics.json").read_text())
    assert m["method"] == "mcdropout" and set(m["score_names"]) == scores
    assert set(m["test_unknown"]["scores"]) == scores
    # tau accepts 95 % of the validation known windows by construction
    val = preds.read_preds(out_dir, "val")
    assert (val["mi"] <= tau["mi"]).mean() == pytest.approx(0.95, abs=0.03)
    # same windows as the softmax run's evaluation
    for s in SETS:
        assert list(preds.read_preds(out_dir, s)["window_id"]) == list(
            preds.read_preds(softmax_run, s)["window_id"]
        )

    # a positional run_dir together with --from, or neither, is a usage error
    bad = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--method", "mcdropout"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert bad.returncode != 0
