"""Figures F2-F5 of evaluation.md §7 from saved predictions (never from models).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

Every function takes plain arrays keyed by a series label, writes the figure to ``out_path``
(format from the suffix) and returns the :class:`matplotlib.figure.Figure`. Figures are built
with ``matplotlib.figure.Figure`` directly, not ``pyplot``, so no GUI backend is touched and
nothing accumulates in pyplot's figure registry.

Series labels are method names, optionally followed by a score after a space, ``:`` or ``/``
(``"softmax"``, ``"edl: vacuity"``). The method part fixes the colour and line style, so a
method looks the same in every figure. Colours are the Okabe-Ito colourblind-safe set (checked
with a CVD validator: worst adjacent deuteranopia ΔE 11); line styles are a second encoding so
the curves stay distinguishable in greyscale. Text sizes are set for slides.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from uaar.eval import calibration, metrics

METHOD_COLORS: dict[str, str] = {
    "logreg": "#E69F00",  # orange
    "softmax": "#0072B2",  # blue
    "edl": "#D55E00",  # vermillion
    "mcdropout": "#009E73",  # bluish green
}
METHOD_LINESTYLES: dict[str, str | tuple] = {
    "logreg": (0, (1, 1.5)),  # dotted
    "softmax": "-",
    "edl": "--",
    "mcdropout": "-.",
}
_FALLBACK_COLORS = ("#CC79A7", "#56B4E9", "#000000", "#999999")
KNOWN_COLOR = "#8C8C8C"  # neutral grey for the known-class reference distribution

STYLE: dict[str, object] = {
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def method_of(label: str) -> str:
    """The method part of a series label: the text before the first space, ``:`` or ``/``."""
    return re.split(r"[\s:/]", label.strip(), maxsplit=1)[0].lower()


def method_color(label: str) -> str:
    """Fixed colour for the label's method; unlisted methods get a stable fallback colour."""
    m = method_of(label)
    if m in METHOD_COLORS:
        return METHOD_COLORS[m]
    return _FALLBACK_COLORS[sum(map(ord, m)) % len(_FALLBACK_COLORS)]


def method_linestyle(label: str):
    """Fixed line style for the label's method (solid for unlisted methods)."""
    return METHOD_LINESTYLES.get(method_of(label), "-")


def _panels(n: int, width: float = 4.6, height: float = 4.0) -> tuple[Figure, list]:
    fig = Figure(figsize=(max(1, n) * width, height), layout="constrained")
    axes = fig.subplots(1, max(1, n), squeeze=False)[0]
    return fig, list(axes)


def _save(fig: Figure, out_path: Path) -> Figure:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    return fig


# ---- F2 ------------------------------------------------------------------------------------
def score_histograms(
    data: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    taus: Mapping[str, float] | None = None,
    bins: int = 40,
    xlabel: str = "uncertainty score",
) -> Figure:
    """F2: score distributions, known vs unknown, one panel per series in a row.

    Args:
        data: ``{series_label: {group_name: scores (n,)}}``. A group called ``"known"`` (or
            ``"test_known"``) is drawn as a filled grey reference; every other group (such as
            ``"unknown"``, ``"non-exercise"``) as an outline in the method colour with its own
            line style.
        out_path: Image path; parent directories are created.
        taus: Optional ``{series_label: tau}``; drawn as a vertical line.
        bins: Histogram bins per panel (shared edges within a panel).
        xlabel: x-axis label.

    Returns:
        The Figure. Histograms are densities so groups of different sizes compare.
    """
    group_styles = ["-", "--", ":", "-."]
    with mpl.rc_context(STYLE):
        fig, axes = _panels(len(data))
        for ax, (label, groups) in zip(axes, data.items(), strict=False):
            arrays = {g: np.asarray(v, dtype=np.float64).ravel() for g, v in groups.items()}
            allv = np.concatenate([v for v in arrays.values() if v.size] or [np.zeros(1)])
            edges = np.histogram_bin_edges(allv, bins=bins)
            color = method_color(label)
            k = 0
            for g, v in arrays.items():
                if v.size == 0:
                    continue
                if g in ("known", "test_known"):
                    ax.hist(v, edges, density=True, color=KNOWN_COLOR, alpha=0.45, label=g)
                else:
                    ax.hist(
                        v,
                        edges,
                        density=True,
                        histtype="step",
                        color=color,
                        linestyle=group_styles[k % len(group_styles)],
                        linewidth=2.2,
                        label=g,
                    )
                    k += 1
            if taus and label in taus:
                ax.axvline(taus[label], color="black", linewidth=1.5, linestyle="--")
                ax.annotate(
                    "τ",
                    (taus[label], 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -16),
                    textcoords="offset points",
                )
            ax.set_title(label)
            ax.set_xlabel(xlabel)
            ax.legend(loc="upper center")
        axes[0].set_ylabel("density")
        return _save(fig, out_path)


# ---- F3 ------------------------------------------------------------------------------------
def roc_curves(
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str = "Unknown detection",
) -> Figure:
    """F3: ROC curves for unknown detection (unknown = positive), all series overlaid.

    Args:
        data: ``{series_label: (score_known (n_k,), score_unknown (n_u,))}``.
        out_path: Image path.
        title: Axes title.

    Returns:
        The Figure; the legend carries each series' AUROC. Series with an empty group are
        listed in the legend as ``n/a`` and not drawn.
    """
    with mpl.rc_context(STYLE):
        fig, (ax,) = _panels(1, width=5.6, height=5.0)
        ax.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=1, linestyle=":")
        for label, (sk, su) in data.items():
            fpr, tpr = metrics.roc_points(sk, su)
            auc = metrics.unknown_detection(sk, su)["auroc"]
            text = f"{label} (AUROC {auc:.3f})" if np.isfinite(auc) else f"{label} (n/a)"
            ax.plot(
                fpr,
                tpr,
                color=method_color(label),
                linestyle=method_linestyle(label),
                label=text,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.01),
            xlabel="known called unknown (FPR)",
            ylabel="unknown detected (TPR)",
            title=title,
            aspect="equal",
        )
        ax.legend(loc="lower right")
        return _save(fig, out_path)


# ---- F4 ------------------------------------------------------------------------------------
def reliability_diagrams(
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    n_bins: int = 15,
) -> Figure:
    """F4: reliability diagrams on known-class windows, one panel per series in a row.

    Args:
        data: ``{series_label: (probs (N, K), y (N,))}``; ``y < 0`` rows are ignored.
        out_path: Image path.
        n_bins: Equal-width confidence bins (15 to match the ECE).

    Returns:
        The Figure. Bars are the accuracy per bin (method colour); the diagonal is perfect
        calibration; the title carries the ECE.
    """
    with mpl.rc_context(STYLE):
        fig, axes = _panels(len(data), width=4.2, height=4.4)
        for ax, (label, (probs, y)) in zip(axes, data.items(), strict=False):
            b = calibration.reliability_bins(probs, y, n_bins)
            e = calibration.ece(probs, y, n_bins)
            centers = (b["edges"][:-1] + b["edges"][1:]) / 2
            width = np.diff(b["edges"])
            used = b["count"] > 0
            ax.bar(
                centers[used],
                b["accuracy"][used],
                width=width[used] * 0.92,
                color=method_color(label),
                alpha=0.85,
                label="accuracy",
            )
            ax.plot(
                b["confidence"][used],
                b["confidence"][used],
                linestyle="none",
                marker="o",
                markersize=5,
                color="black",
                label="mean confidence",
            )
            ax.plot([0, 1], [0, 1], color="#555555", linewidth=1.2, linestyle="--")
            ece_txt = f"ECE {e:.3f}" if np.isfinite(e) else "ECE n/a"
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="confidence (max p)",
                title=f"{label}\n{ece_txt}",
                aspect="equal",
            )
        axes[0].set_ylabel("accuracy")
        axes[0].legend(loc="upper left")
        return _save(fig, out_path)


# ---- F5 ------------------------------------------------------------------------------------
def risk_coverage_curves(
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str = "Selective prediction",
) -> Figure:
    """F5: risk-coverage curves on known-class windows, all series overlaid.

    Args:
        data: ``{series_label: (score (N,), correct (N,) bool)}``.
        out_path: Image path.
        title: Axes title.

    Returns:
        The Figure; the legend carries each series' AURC (lower is better).
    """
    with mpl.rc_context(STYLE):
        fig, (ax,) = _panels(1, width=6.0, height=4.6)
        for label, (score, correct) in data.items():
            cov, risk, aurc = metrics.risk_coverage(score, correct)
            text = f"{label} (AURC {aurc:.3f})" if np.isfinite(aurc) else f"{label} (n/a)"
            ax.plot(
                cov,
                risk,
                color=method_color(label),
                linestyle=method_linestyle(label),
                label=text,
            )
        ax.set(xlim=(0, 1), xlabel="coverage (fraction kept)", ylabel="risk (error rate)")
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, out_path)
