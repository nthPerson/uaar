"""uaar.eval.plots: each figure function writes a non-empty file (Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from uaar.eval import plots  # noqa: E402

METHODS = ("softmax", "edl", "mcdropout", "logreg")


@pytest.fixture(scope="module")
def fake():
    rng = np.random.default_rng(0)
    out = {}
    for i, m in enumerate(METHODS):
        logits = rng.normal(0, 1 + i, (400, 5))
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y = rng.integers(0, 5, 400)
        score = 1 - probs.max(1)
        out[m] = {
            "known": rng.normal(0, 1, 400),
            "unknown": rng.normal(1 + 0.3 * i, 1, 200),
            "nonex": rng.normal(0.5, 1, 100),
            "probs": probs,
            "y": y,
            "score": score,
            "correct": probs.argmax(1) == y,
        }
    return out


def _check(fig, path):
    assert isinstance(fig, Figure)
    assert path.exists() and path.stat().st_size > 1000


def test_score_histograms(fake, tmp_path):
    data = {m: {k: fake[m][k] for k in ("known", "unknown", "nonex")} for m in METHODS}
    path = tmp_path / "f2.png"
    _check(plots.score_histograms(data, path, taus={"edl": 1.5}), path)


def test_roc_curves(fake, tmp_path):
    data = {f"{m}: s": (fake[m]["known"], fake[m]["unknown"]) for m in METHODS}
    data["edl: empty"] = (fake["edl"]["known"], np.array([]))  # drawn as n/a, no crash
    path = tmp_path / "sub" / "f3.pdf"
    _check(plots.roc_curves(data, path), path)


def test_reliability_diagrams(fake, tmp_path):
    data = {m: (fake[m]["probs"], fake[m]["y"]) for m in METHODS}
    path = tmp_path / "f4.png"
    _check(plots.reliability_diagrams(data, path), path)


def test_risk_coverage_curves(fake, tmp_path):
    data = {m: (fake[m]["score"], fake[m]["correct"]) for m in METHODS}
    path = tmp_path / "f5.svg"
    _check(plots.risk_coverage_curves(data, path), path)


def test_method_colours_are_fixed():
    assert plots.method_color("edl") == plots.method_color("EDL: vacuity")
    assert plots.method_color("softmax/msp") == plots.METHOD_COLORS["softmax"]
    assert len(set(plots.METHOD_COLORS.values())) == len(plots.METHOD_COLORS)
    assert plots.method_color("newmethod") == plots.method_color("newmethod: x")
