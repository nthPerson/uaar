import numpy as np

from uaar.models import normalize


def test_fit_apply_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(50, 6, 100)) * np.arange(1, 7)[None, :, None] + 3).astype(np.float32)
    stats = normalize.fit_stats(X)
    Z = normalize.apply(X, stats)
    assert Z.dtype == np.float32 and Z.shape == X.shape
    assert np.allclose(Z.mean(axis=(0, 2)), 0, atol=1e-3) and np.allclose(
        Z.std(axis=(0, 2)), 1, atol=1e-3
    )
    normalize.save(stats, tmp_path / "preprocess.json", window_samples=100, sample_rate_hz=50)
    loaded = normalize.load(tmp_path / "preprocess.json")
    assert loaded["mean"] == stats["mean"] and loaded["window_samples"] == 100
