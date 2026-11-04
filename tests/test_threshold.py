import numpy as np
import pytest

from uaar.eval import threshold


def test_select_tau_accepts_95_percent():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=10_000)
    tau = threshold.select_tau(s, accept=0.95)
    assert (~threshold.apply(s, tau)).mean() == pytest.approx(0.95, abs=0.001)
    with pytest.raises(ValueError):
        threshold.select_tau(np.array([]))
