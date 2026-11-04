"""Backbone invariants (spec §7): output shape, parameter budget, MAC count."""

import torch

from uaar.models.backbone import ConvBackbone


def test_output_shape_and_param_budget():
    net = ConvBackbone(n_classes=12)
    x = torch.randn(4, 6, 100)
    assert net(x).shape == (4, 12)
    assert net.embed(x).shape == (4, 64)
    n = net.count_params()
    assert abs(n - 30_000) / 30_000 < 0.10, n  # within 10 % of 30 k (K = 12)


def test_macs_positive_and_stable():
    net = ConvBackbone(n_classes=12)
    macs = net.count_macs(100)
    assert macs > 0
    assert macs == net.count_macs(100)
    assert net.count_macs(40) < macs  # RecGym windows are shorter
    # hand count: conv1 32*100*6*5 + conv2 64*50*32*5 + conv3 64*25*64*3 + fc 64*64 + head 64*12
    assert macs == 32 * 100 * 30 + 64 * 50 * 160 + 64 * 25 * 192 + 64 * 64 + 64 * 12


def test_eval_mode_is_deterministic_and_train_mode_is_not():
    torch.manual_seed(0)
    net = ConvBackbone(n_classes=5, dropout=0.5)
    x = torch.randn(8, 6, 100)
    net.eval()
    assert torch.allclose(net(x), net(x))
    net.train()
    assert not torch.allclose(net(x), net(x))
