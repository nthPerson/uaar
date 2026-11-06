"""EDL invariants (spec §7 test_edl)."""

import math

import pytest
import torch
import torch.nn.functional as F

from uaar.models import edl


def test_kl_of_uniform_is_zero():
    alpha = torch.ones(3, 7, dtype=torch.float64)
    assert torch.allclose(
        edl.kl_dirichlet_uniform(alpha), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )
    # and positive otherwise
    assert (
        edl.kl_dirichlet_uniform(torch.tensor([[5.0, 1.0, 1.0]], dtype=torch.float64)) > 0
    ).all()


def test_loss_finite_for_extreme_evidence():
    y = F.one_hot(torch.tensor([0, 1]), 3).double()
    for scale in (1e-6, 1.0, 1e6):
        alpha = 1.0 + scale * torch.rand(2, 3, dtype=torch.float64)
        for fn in (edl.edl_loss, edl.edl_loss_digamma):
            loss = fn(alpha, y, lam=1.0)
            assert torch.isfinite(loss), (fn.__name__, scale)


def test_gradcheck_tiny_batch():
    torch.manual_seed(0)
    y = F.one_hot(torch.tensor([2, 0]), 3).double()
    z = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    def f(z):
        return edl.edl_loss(edl.evidence_from_logits(z) + 1.0, y, lam=0.5)

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-5)


def test_mse_and_digamma_variants_agree_in_ordering():
    # three samples: confident-correct, uncertain, confident-wrong (true class 0)
    y = F.one_hot(torch.tensor([0, 0, 0]), 3).double()
    alpha = torch.tensor([[50.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 50.0, 1.0]], dtype=torch.float64)
    a = edl.edl_loss(alpha, y, lam=1.0, reduction="none")
    b = edl.edl_loss_digamma(alpha, y, lam=1.0, reduction="none")
    c = edl.edl_loss_log(alpha, y, lam=1.0, reduction="none")
    assert torch.argsort(a).tolist() == [0, 1, 2]
    assert torch.equal(torch.argsort(a), torch.argsort(b))
    assert torch.equal(torch.argsort(a), torch.argsort(c))


def test_log_loss_hand_case_and_zero_kl_weight():
    y = F.one_hot(torch.tensor([1]), 3).double()
    alpha = torch.tensor([[1.0, 6.0, 3.0]], dtype=torch.float64)
    got = edl.edl_loss_log(alpha, y, lam=0.0)
    assert got.item() == pytest.approx(math.log(10.0) - math.log(6.0))


def test_method_options_are_checked_saved_and_loaded(tmp_path):
    with pytest.raises(ValueError, match="unknown EDL loss"):
        edl.EDLMethod(3, ["a", "b", "c"], loss="nope")
    m = edl.EDLMethod(3, ["a", "b", "c"], loss="log", kl_weight=0.1, kl_anneal_epochs=5)
    m.stats = {"mean": [0.0] * 6, "std": [1.0] * 6}
    m.save(tmp_path)
    back = edl.EDLMethod.load(tmp_path)
    assert (back.loss_kind, back.kl_weight, back.kl_anneal_epochs) == ("log", 0.1, 5)
    assert edl.EDLMethod(3, ["a", "b", "c"]).loss_kind == "digamma"


def test_vacuity_range_and_monotone_in_evidence():
    K = 5
    for total_evidence in (0.0, 1.0, 10.0, 1000.0):
        alpha = torch.ones(1, K) + total_evidence / K
        u = edl.dirichlet_summary(alpha)["vacuity"]
        assert 0 < u.item() <= 1.0
    us = [
        edl.dirichlet_summary(torch.ones(1, K) + e / K)["vacuity"].item()
        for e in (0.0, 1.0, 10.0, 1000.0)
    ]
    assert us[0] == 1.0 and all(a > b for a, b in zip(us, us[1:], strict=False))


def test_class_weights_scale_per_sample_loss():
    y = F.one_hot(torch.tensor([0, 1]), 2).double()
    alpha = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    base = edl.edl_loss(alpha, y, lam=0.0, reduction="none")
    weighted = edl.edl_loss(
        alpha,
        y,
        lam=0.0,
        class_weights=torch.tensor([2.0, 1.0], dtype=torch.float64),
        reduction="none",
    )
    assert torch.allclose(weighted, base * torch.tensor([2.0, 1.0], dtype=torch.float64))


def test_edlnet_outputs_alpha_above_one():
    net = edl.EDLNet(n_classes=4).eval()
    alpha = net(torch.randn(3, 6, 100))
    assert alpha.shape == (3, 4) and (alpha >= 1.0).all()
    s = edl.dirichlet_summary(alpha)
    assert torch.allclose(s["probs"].sum(dim=1), torch.ones(3))
    assert edl.kl_anneal(0) == 0.0 and edl.kl_anneal(5) == 0.5 and edl.kl_anneal(20) == 1.0
