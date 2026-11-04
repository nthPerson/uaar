"""EDL invariants (spec §7 test_edl)."""

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
    assert torch.equal(torch.argsort(a), torch.argsort(b)) and torch.argsort(a).tolist() == [
        0,
        1,
        2,
    ]


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
