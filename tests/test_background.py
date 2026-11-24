"""Background windows in training: the taught losses, the Other class, and the three tiers end
to end on a tiny synthetic dataset (decisions.md 2026-11-23)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml

from uaar.config import Config
from uaar.data import splits
from uaar.models import edl, factory
from uaar.models.base import with_background
from uaar.models.edl import EDLMethod
from uaar.models.logreg import LogRegMethod
from uaar.models.mcdropout import MCDropoutMethod
from uaar.models.softmax import SoftmaxMethod, taught_cross_entropy, uniform_cross_entropy

K = 3


# ---- Loss terms -------------------------------------------------------------------------------
def test_uniform_cross_entropy_is_smallest_for_a_uniform_prediction():
    flat = torch.full((1, K), 2.5, dtype=torch.float64, requires_grad=True)
    loss = uniform_cross_entropy(flat)
    assert loss.item() == pytest.approx(math.log(K))
    (grad,) = torch.autograd.grad(loss.sum(), flat)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-12)  # a stationary point
    rng = torch.Generator().manual_seed(0)
    other = torch.randn(200, K, generator=rng, dtype=torch.float64) * 3
    assert (uniform_cross_entropy(other) > math.log(K)).all()  # and it is the minimum


def test_taught_cross_entropy_is_the_sum_of_its_two_terms():
    rng = torch.Generator().manual_seed(1)
    z = torch.randn(10, K, generator=rng)
    y = torch.tensor([0, 1, 2, K, K, 0, K, 1, 2, K])
    w = torch.tensor([1.0, 2.0, 0.5])
    known = y < K
    expected = (
        F.cross_entropy(z[known], y[known], weight=w)
        + 0.3 * uniform_cross_entropy(z[~known]).mean()
    )
    assert taught_cross_entropy(z, y, K, w, 0.3).item() == pytest.approx(expected.item())
    # no background rows: ordinary cross-entropy; only background rows: only that term
    assert taught_cross_entropy(z[known], y[known], K, w, 0.3).item() == pytest.approx(
        F.cross_entropy(z[known], y[known], weight=w).item()
    )
    assert taught_cross_entropy(z[~known], y[~known], K, w, 1.0).item() == pytest.approx(
        uniform_cross_entropy(z[~known]).mean().item()
    )


def test_taught_edl_loss_background_term_is_zero_only_without_evidence():
    y_bg = torch.full((4,), K)
    no_evidence = torch.ones(4, K, dtype=torch.float64, requires_grad=True)
    loss = edl.taught_edl_loss(no_evidence, y_bg, K, lam=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    (grad,) = torch.autograd.grad(loss, no_evidence)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-9)  # the minimum: alpha = 1
    some = torch.tensor([[1.0, 3.0, 1.0], [2.0, 2.0, 2.0], [9.0, 1.0, 1.0], [1.0, 1.0, 1.5]])
    assert edl.taught_edl_loss(some.double(), y_bg, K, lam=1.0).item() > 0
    # more evidence on a background row costs more
    a = edl.taught_edl_loss(torch.tensor([[2.0, 1.0, 1.0]]).double(), y_bg[:1], K, lam=1.0)
    b = edl.taught_edl_loss(torch.tensor([[20.0, 1.0, 1.0]]).double(), y_bg[:1], K, lam=1.0)
    assert b.item() > a.item() > 0


def test_taught_edl_loss_known_rows_are_the_ordinary_loss():
    alpha = torch.tensor([[5.0, 1.0, 2.0], [1.0, 1.0, 7.0], [3.0, 3.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([0, 2, K])
    w = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y1 = F.one_hot(y[:2], K).double()
    known = edl.edl_loss_digamma(alpha[:2], y1, lam=0.4, class_weights=w)
    bg = edl.kl_dirichlet_uniform(alpha[2:]).mean()
    got = edl.taught_edl_loss(alpha, y, K, lam=0.4, class_weights=w, background_weight=2.0)
    assert got.item() == pytest.approx((known + 2.0 * bg).item())
    assert edl.taught_edl_loss(alpha[:2], y[:2], K, lam=0.4, class_weights=w).item() == (
        pytest.approx(known.item())
    )


# ---- Methods on synthetic windows ----------------------------------------------------------------
def _windows(freqs, n_each, seed):
    """Sine windows, one frequency per class; returns X (N, 6, 100) and the class index."""
    rng = np.random.default_rng(seed)
    t = np.arange(100) / 50
    X, y = [], []
    for k, f in enumerate(freqs):
        for _ in range(n_each):
            sig = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            cos = 50 * np.cos(2 * np.pi * f * t)
            X.append(np.stack([sig, 0.3 * sig, np.ones(100), cos, np.zeros(100), 5 * sig]))
            y.append(k)
    X = np.asarray(X, np.float32) + rng.normal(0, 0.05, (len(X), 6, 100)).astype(np.float32)
    return X, np.asarray(y)


KNOWN_F, BG_F, NEW_F = (0.5, 1.5, 2.5), (4.0, 6.0, 8.0), (5.0, 7.0)
CLASSES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    Xtr, ytr = _windows(KNOWN_F, 60, 0)
    Xva, yva = _windows(KNOWN_F, 20, 1)
    Xte, yte = _windows(KNOWN_F, 30, 2)
    bg_tr, _ = _windows(BG_F, 60, 3)
    bg_va, _ = _windows(BG_F, 20, 4)
    new, _ = _windows(NEW_F, 40, 5)  # frequencies between the background ones, never trained on
    return Xtr, ytr, Xva, yva, Xte, yte, (bg_tr, bg_va), new


def _cfg(method, mode):
    return Config.load(
        f"configs/p1/{method}.yaml",
        [f"background.mode={mode}", "train.epochs=25", "train.batch_size=32", "train.patience=25"],
    )


def test_with_background_appends_marker_rows():
    X, y = with_background(np.zeros((3, 6, 100)), np.array([0, 1, 2]), np.ones((2, 6, 100)), K)
    assert X.shape == (5, 6, 100) and X.dtype == np.float32
    assert y.tolist() == [0, 1, 2, K, K]


@pytest.mark.parametrize(
    ("method", "mode", "score"),
    [("softmax", "uncertain", "msp"), ("edl", "uncertain", "vacuity"), ("softmax", "other", "msp")],
)
def test_background_modes_reject_new_windows_and_round_trip(tmp_path, data, method, mode, score):
    Xtr, ytr, Xva, yva, Xte, yte, bg, new = data
    cfg = _cfg(method, mode)
    torch.manual_seed(0)
    m = factory.build(cfg, CLASSES, seed=0)
    assert m.background == mode
    with pytest.raises(ValueError, match="needs background windows"):
        m.fit(Xtr, ytr, Xva, yva, cfg, tmp_path)
    m.fit(Xtr, ytr, Xva, yva, cfg, tmp_path, background=bg)

    known, unseen = m.predict(Xte), m.predict(new)
    assert known.probs.shape == (len(Xte), K) and set(known.scores) == set(m.score_names)
    assert (known.label == yte).mean() > 0.95
    # the never-seen frequencies are more uncertain than the known ones: most of them lie
    # above the 95th percentile of the known validation scores
    tau = np.quantile(m.predict(Xva).scores[score], 0.95)
    assert (unseen.scores[score] > tau).mean() > 0.8
    assert (known.scores[score] <= tau).mean() > 0.8
    if mode == "other":
        assert m.score_names == ["msp", "p_other"]
        total = known.probs.sum(axis=1) + known.scores["p_other"]
        np.testing.assert_allclose(total, 1.0, atol=1e-5)
        assert unseen.scores["p_other"].mean() > 0.5

    m.save(tmp_path)
    conf = json.loads((tmp_path / "config.json").read_text())
    assert conf["background"] == mode and conf["classes"] == CLASSES
    back = factory.load(tmp_path)
    assert back.background == mode and back.score_names == m.score_names
    again = back.predict(new)
    np.testing.assert_allclose(again.probs, unseen.probs, atol=1e-6)
    np.testing.assert_allclose(again.scores[score], unseen.scores[score], atol=1e-6)

    if method == "softmax" and mode == "uncertain":
        # MC Dropout on the taught network: the passes agree on the flat output, so the
        # entropy of the mean rejects the new windows while the mutual information stays small
        mc = MCDropoutMethod.from_softmax(m, T=10)
        p = mc.predict(new)
        assert p.scores["entropy"].mean() > 0.8 * math.log(K)
        assert p.scores["mi"].mean() < 0.1 * p.scores["entropy"].mean()
    if mode == "other":
        with pytest.raises(ValueError, match="not defined"):
            MCDropoutMethod.from_softmax(m)


def test_plain_models_ignore_background_and_modes_are_validated(tmp_path, data):
    Xtr, ytr, Xva, yva, _, _, bg, _ = data
    cfg = _cfg("softmax", "none")
    cfg.set("train.epochs", 2)
    torch.manual_seed(0)
    a = factory.build(cfg, CLASSES, seed=0)
    a.fit(Xtr, ytr, Xva, yva, cfg, tmp_path)
    torch.manual_seed(0)
    b = factory.build(cfg, CLASSES, seed=0)
    b.fit(Xtr, ytr, Xva, yva, cfg, tmp_path, background=bg)
    np.testing.assert_allclose(a.predict(Xva).probs, b.predict(Xva).probs, atol=1e-6)

    with pytest.raises(ValueError, match="background must be one of"):
        SoftmaxMethod(K, CLASSES, background="nope")
    with pytest.raises(ValueError, match="'none' or 'uncertain'"):
        EDLMethod(K, CLASSES, background="other")
    with pytest.raises(ValueError, match="plain baseline"):
        factory.build(_cfg("logreg", "uncertain"), CLASSES)
    assert isinstance(factory.build(_cfg("logreg", "none"), CLASSES), LogRegMethod)


def test_primary_score_depends_on_the_background_mode():
    assert factory.primary_score("mcdropout") == factory.primary_score("mcdropout", "none") == "mi"
    assert factory.primary_score("mcdropout", "uncertain") == "entropy"
    for method in ("softmax", "edl", "logreg"):
        for mode in (None, "none", "uncertain", "other"):
            assert factory.primary_score(method, mode) == factory.PRIMARY_SCORE[method]


def test_tier_configs_resolve():
    expect = {
        "k4plain": ("none", ["softmax", "edl", "logreg", "mcdropout"]),
        "k4taught": ("uncertain", ["softmax", "edl", "mcdropout"]),
        "k4other": ("other", ["softmax"]),
    }
    for protocol, (mode, methods) in expect.items():
        for method in methods:
            cfg = Config.load(f"configs/{protocol}/{method}.yaml")
            assert cfg.protocol == protocol and cfg.method == method
            if method != "mcdropout":  # MC Dropout takes the mode from its softmax run
                assert cfg.get("background.mode") == mode
    assert Config.load("configs/k4taught/softmax.yaml").get("background.weight") == 0.5
    assert Config.load("configs/k4taught/edl.yaml").get("background.weight") == 1.0
    assert Config.load("configs/p1/softmax.yaml").get("background.mode") == "none"
    sweep = yaml.safe_load(Path("configs/sweeps/k4.yaml").read_text())
    assert {p: sorted(m) for p, m in sweep["protocols"].items()} == {
        p: sorted(m) for p, (_, m) in expect.items()
    }


# ---- The three tiers through the scripts ---------------------------------------------------------
def _make_dataset(root: Path, n_subjects=12, per=6):
    """Eight sine classes (class0..class7) in "single" recordings and non-exercise noise in
    "multi" recordings; bg0 split with two known classes and one look-alike."""
    rng = np.random.default_rng(0)
    t = np.arange(100) / 50
    X, rows = [], []
    for s in range(n_subjects):
        for k in range(8):
            for i in range(per):
                sig = np.sin(2 * np.pi * (0.5 + k) * t + rng.uniform(0, 6.28))
                cos = 50 * np.cos(2 * np.pi * (0.5 + k) * t)
                X.append(np.stack([sig, 0.3 * sig, np.ones(100), cos, np.zeros(100), 5 * sig]))
                rows.append((f"S{s:02d}", f"single/S{s:02d}/{k}", "single", f"class{k}", float(i)))
        for i in range(per):
            X.append(rng.normal(0, 0.3, (6, 100)))
            rows.append((f"S{s:02d}", f"multi/S{s:02d}", "multi", "Rest", float(i)))
    X = np.asarray(X, np.float32) + rng.normal(0, 0.05, (len(X), 6, 100)).astype(np.float32)
    meta = pd.DataFrame(rows, columns=["subject", "recording_id", "source", "label", "start_s"])
    meta.insert(0, "window_id", np.arange(len(meta)))
    meta["label_purity"] = 1.0
    meta["n_gaps"] = 0
    d = root / "processed" / "recofit" / "windows"
    d.mkdir(parents=True)
    np.save(d / "X.npy", X)
    meta.to_parquet(d / "meta.parquet", index=False)
    splits.write_background_splits(
        meta,
        root / "processed" / "recofit" / "splits",
        known=["class0", "class1"],
        lookalike=["class2"],
        seeds=(0,),
        n_folds=3,
        min_subjects=2,
        min_minutes=0.1,
        dataset="recofit",
        exclude=["Rest"],
        class_source="single",
    )


def test_three_tiers_through_sweep_evaluate_and_tables(tmp_path):
    _make_dataset(tmp_path / "data")
    env = {**os.environ, "UAAR_DATA_DIR": str(tmp_path / "data")}
    sweep = tmp_path / "sweep.yaml"
    protocols = {
        "k4plain": ["softmax"],
        "k4taught": ["softmax", "mcdropout", "edl"],
        "k4other": ["softmax"],
    }
    sweep.write_text(
        yaml.safe_dump(
            {"folds": [0], "class_splits": ["bg0"], "seeds": [0], "protocols": protocols}
        )
    )
    runs = tmp_path / "runs"
    cmd = [
        sys.executable, "scripts/sweep.py", str(sweep), "--runs-dir", str(runs),
        "--set", "train.epochs=4", "train.batch_size=32", "model.mcdropout.T=3",
        "background.n_exercise=60", "background.n_nonex=30",
        "background.n_val_exercise=20", "background.n_val_nonex=10",
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "5 run, 0 skipped, 0 failed" in out.stdout
    # 60 background-class windows and 30 non-exercise windows, the caps set above
    assert "background (uncertain): train 90 windows" in out.stdout
    assert "background (other): train 90 windows" in out.stdout

    split = splits.load_split(runs / "k4taught_recofit_edl_fold0_bg0_seed0" / "split.json")
    assert split.known_classes == ["class0", "class1"] and split.lookalike_classes == ["class2"]
    sets = ["test_known", "test_unknown", "test_nonex", "test_background", "test_lookalike"]
    for protocol, mode in (("k4plain", "none"), ("k4taught", "uncertain"), ("k4other", "other")):
        for method in protocols[protocol]:
            run_dir = runs / f"{protocol}_recofit_{method}_fold0_bg0_seed0"
            m = json.loads((run_dir / "metrics.json").read_text())
            assert (m["protocol"], m["method"], m["background"]) == (protocol, method, mode)
            assert m["K"] == 2 and [s for s in sets if s in m] == sets
            for name in sets[1:]:
                assert m[name]["n"] > 0
                for vals in m[name]["scores"].values():
                    assert 0.0 <= vals["accepted"] <= 1.0
                    assert vals["accepted"] == pytest.approx(1.0 - vals["unknown_recall"])
            # the saved probabilities always have one column per known class
            preds = pd.read_parquet(run_dir / "preds_test_unknown.parquet")
            assert len(preds["probs"].iloc[0]) == 2
    other = json.loads(
        (runs / "k4other_recofit_softmax_fold0_bg0_seed0" / "metrics.json").read_text()
    )
    assert other["score_names"] == ["msp", "p_other"]
    # logistic regression has no taught version: train.py refuses the combination
    bad = [
        sys.executable, "scripts/train.py", "--config", "configs/k4plain/logreg.yaml",
        "--fold", "0", "--class-split", "bg0", "--runs-dir", str(runs),
        "--set", "background.mode=uncertain",
    ]  # fmt: skip
    out = subprocess.run(bad, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "plain baseline" in out.stderr

    tables = tmp_path / "tables"
    cmd = [sys.executable, "scripts/make_tables.py", "--runs-dir", str(runs), "--out", str(tables)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    t2 = pd.read_csv(tables / "t2_headline.csv")
    assert list(zip(t2["protocol"], t2["method"], t2["score"], strict=True)) == [
        ("k4plain", "softmax", "msp"),
        ("k4taught", "softmax", "msp"),
        ("k4taught", "mcdropout", "entropy"),  # not mutual information for a taught network
        ("k4taught", "edl", "vacuity"),
        ("k4other", "softmax", "msp"),
    ]
    assert (t2["n_runs"] == 1).all() and t2["accepted_mean"].between(0, 1).all()
    detail = pd.read_csv(tables / "scores_by_protocol.csv")
    mc = detail[(detail["protocol"] == "k4taught") & (detail["method"] == "mcdropout")]
    assert set(mc["score"]) == {"entropy", "mi", "varratio"}
    assert mc.set_index("score")["primary"].to_dict() == {
        "entropy": True,
        "mi": False,
        "varratio": False,
    }
    for col in ("unknown_accepted_mean", "nonex_accepted_mean", "background_accepted_mean",
                "lookalike_accepted_mean", "known_kept_mean"):  # fmt: skip
        assert detail[col].between(0, 1).all(), col
    assert "| k4taught | mcdropout | entropy * |" in (tables / "scores_by_protocol.md").read_text()
