"""Tiny end-to-end training on synthetic windows: softmax and EDL both learn and save artifacts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from uaar.config import Config
from uaar.models import edl
from uaar.models.softmax import SoftmaxMethod
from uaar.models.train_loop import fit, predict_batches


def _synthetic(n_per_class=80, K=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(100) / 50
    X, y = [], []
    for k in range(K):
        f = 0.5 + k  # distinct frequency per class
        for _ in range(n_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            sig = np.sin(2 * np.pi * f * t + phase)
            X.append(
                np.stack(
                    [
                        sig,
                        0.3 * sig,
                        np.ones(100),
                        50 * np.cos(2 * np.pi * f * t),
                        np.zeros(100),
                        5 * sig,
                    ]
                )
                + rng.normal(0, 0.05, (6, 100))
            )
            y.append(k)
    X, y = np.asarray(X, np.float32), np.asarray(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def data():
    X, y = _synthetic()
    return X[:180], y[:180], X[180:], y[180:]


def test_softmax_method_learns_and_round_trips(data, tmp_path):
    torch.manual_seed(0)
    Xtr, ytr, Xval, yval = data
    cfg = Config.load(
        "configs/p1/softmax.yaml", ["train.epochs=6", "train.batch_size=32", "train.patience=3"]
    )
    m = SoftmaxMethod(3, ["a", "b", "c"])
    m.fit(Xtr, ytr, Xval, yval, cfg, tmp_path)
    pred = m.predict(Xval)
    assert pred.probs.shape == (len(yval), 3) and (pred.label == yval).mean() > 0.8
    assert set(pred.scores) == {"msp", "entropy"} and (tmp_path / "history.csv").exists()
    m.save(tmp_path)
    m2 = SoftmaxMethod.load(tmp_path)
    assert np.allclose(m2.predict(Xval).probs, pred.probs, atol=1e-5)


def test_edl_trains_with_annealed_kl(data):
    torch.manual_seed(0)
    Xtr, ytr, Xval, yval = data
    net = edl.EDLNet(3)

    def loss_fn(alpha, y, epoch):
        return edl.edl_loss(alpha, F.one_hot(y, 3).to(alpha.dtype), lam=edl.kl_anneal(epoch, 3))

    hist = fit(
        net,
        loss_fn,
        lambda a: a / a.sum(dim=1, keepdim=True),
        Xtr,
        ytr,
        Xval,
        yval,
        epochs=6,
        batch_size=32,
        patience=3,
        log=None,
        extra_columns=lambda e: {"lambda": edl.kl_anneal(e, 3)},
    )
    assert hist.best_val_f1 > 0.8 and hist.rows[-1]["lambda"] <= 1.0
    s = edl.dirichlet_summary(predict_batches(net, Xval))
    assert 0 < s["vacuity"].mean() < 1
