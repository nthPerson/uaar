"""MC Dropout: mode handling, the T = 1 / p = 0 identity, score ranges, repeatability."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from uaar.models import factory
from uaar.models.mcdropout import MCDropoutMethod, enable_dropout_only, mc_scores
from uaar.models.softmax import SoftmaxMethod, SoftmaxNet

K = 4
STATS = {"mean": [0.0] * 6, "std": [1.0] * 6}
CLASSES = [f"c{k}" for k in range(K)]


def _net(dropout=0.3, seed=0):
    torch.manual_seed(seed)
    net = SoftmaxNet(K, dropout=dropout)
    # give BatchNorm non-trivial running statistics so eval and train mode differ
    net.train()
    with torch.no_grad():
        for _ in range(3):
            net(torch.randn(32, 6, 100) * 2 + 1)
    net.eval()
    return net


def _X(n=50, seed=1):
    return np.random.default_rng(seed).normal(0, 1, (n, 6, 100)).astype(np.float32)


def test_enable_dropout_only_modes():
    net = _net()
    enable_dropout_only(net)
    drops = [m for m in net.modules() if isinstance(m, nn.Dropout)]
    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    assert len(drops) == 2 and len(norms) == 3
    assert all(m.training for m in drops)
    assert not any(m.training for m in norms)


def test_t1_p0_equals_softmax():
    net = _net(dropout=0.0)
    X = _X()
    with torch.no_grad():
        expected = F.softmax(net(torch.from_numpy(X)), dim=1).numpy()
    pred = MCDropoutMethod(net, CLASSES, STATS, T=1).predict(X)
    np.testing.assert_allclose(pred.probs, expected, atol=1e-6)
    np.testing.assert_array_equal(pred.label, expected.argmax(axis=1))
    np.testing.assert_allclose(pred.scores["mi"], 0.0, atol=1e-6)
    np.testing.assert_array_equal(pred.scores["varratio"], 0.0)


def test_score_ranges_and_shapes():
    X = _X(70)
    pred = MCDropoutMethod(_net(dropout=0.5), CLASSES, STATS, T=20).predict(X, batch_size=32)
    assert pred.probs.shape == (70, K) and pred.label.shape == (70,)
    np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(pred.scores) == {"entropy", "mi", "varratio"}
    for s in pred.scores.values():
        assert s.shape == (70,) and np.isfinite(s).all()
    assert (pred.scores["mi"] >= 0).all()
    assert (pred.scores["mi"] <= pred.scores["entropy"] + 1e-9).all()
    assert ((pred.scores["varratio"] >= 0) & (pred.scores["varratio"] <= 1 - 1 / K)).all()
    assert (pred.scores["entropy"] <= np.log(K) + 1e-6).all()
    assert pred.scores["mi"].max() > 0  # dropout is really on


def test_repeatable_and_leaves_global_rng_alone():
    net, X = _net(), _X()
    m = MCDropoutMethod(net, CLASSES, STATS, T=10, seed=3)
    torch.manual_seed(123)
    a = m.predict(X)
    after_first = torch.rand(1).item()
    torch.manual_seed(123)
    expected_next = torch.rand(1).item()
    assert after_first == expected_next  # predict did not advance or reseed the global stream
    b = m.predict(X)
    np.testing.assert_array_equal(a.probs, b.probs)
    c = MCDropoutMethod(net, CLASSES, STATS, T=10, seed=4).predict(X)
    assert not np.array_equal(a.probs, c.probs)
    assert not net.training  # the network is handed back in the mode it came in


def test_shared_trunk_equals_full_passes():
    net, X = _net(), _X()
    fast = MCDropoutMethod(net, CLASSES, STATS, T=8, share_trunk=True).predict(X, batch_size=16)
    full = MCDropoutMethod(net, CLASSES, STATS, T=8, share_trunk=False).predict(X, batch_size=16)
    np.testing.assert_allclose(fast.probs, full.probs, atol=1e-6)
    np.testing.assert_allclose(fast.scores["mi"], full.scores["mi"], atol=1e-6)
    np.testing.assert_array_equal(fast.scores["varratio"], full.scores["varratio"])


def test_batchnorm_statistics_not_updated():
    net = _net()
    before = [m.running_mean.clone() for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    MCDropoutMethod(net, CLASSES, STATS, T=5).predict(_X() * 5 + 3)
    after = [m.running_mean for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    for b, a in zip(before, after, strict=True):
        torch.testing.assert_close(b, a)


def test_mc_scores_hand_case():
    # two passes that disagree completely on a 2-class problem
    prob_sum = np.array([[1.0, 1.0], [2.0, 0.0]])
    ent_sum = np.array([0.0, 0.0])
    votes = np.array([[1, 1], [2, 0]])
    s = mc_scores(prob_sum, ent_sum, votes, T=2)
    np.testing.assert_allclose(s["probs"], [[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(s["entropy"], [np.log(2), 0.0], atol=1e-9)
    np.testing.assert_allclose(s["mi"], [np.log(2), 0.0], atol=1e-9)
    np.testing.assert_allclose(s["varratio"], [0.5, 0.0])


def test_from_softmax_run_save_load_and_factory(tmp_path):
    sm = SoftmaxMethod(K, CLASSES, dropout=0.3)
    sm.model = _net()
    sm.stats = {"mean": [0.1] * 6, "std": [2.0] * 6}
    sm.save(tmp_path / "p1_x_softmax_fold0_rand0_seed0")
    m = MCDropoutMethod.from_softmax_run(tmp_path / "p1_x_softmax_fold0_rand0_seed0", T=7, seed=2)
    assert m.source_run == "p1_x_softmax_fold0_rand0_seed0"
    X = _X()
    a = m.predict(X)
    m.save(tmp_path / "mc")
    for loaded in (MCDropoutMethod.load(tmp_path / "mc"), factory.load(tmp_path / "mc")):
        assert isinstance(loaded, MCDropoutMethod) and loaded.T == 7 and loaded.seed == 2
        b = loaded.predict(X)
        np.testing.assert_allclose(a.probs, b.probs, atol=1e-6)
        np.testing.assert_allclose(a.scores["entropy"], b.scores["entropy"], atol=1e-6)


def test_fit_is_an_error():
    m = MCDropoutMethod(_net(), CLASSES, STATS)
    with pytest.raises(RuntimeError, match="does not train"):
        m.fit(None, None, None, None, None, None)
    with pytest.raises(ValueError):
        MCDropoutMethod(_net(), CLASSES, STATS, T=0)
