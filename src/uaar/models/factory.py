"""Build a Method from a resolved Config (development-spec §4.3)."""

from __future__ import annotations

from pathlib import Path

from uaar.config import Config

#: The one uncertainty score per method that the headline table and the live demo show.
PRIMARY_SCORE: dict[str, str] = {
    "softmax": "msp",
    "logreg": "msp",
    "edl": "vacuity",
    "mcdropout": "mi",
}

#: Exceptions by ``background.mode``. A network taught to be uncertain on background windows
#: gives the same flat output in every dropout pass, so the passes agree and the mutual
#: information is near zero for exactly the windows that should be rejected; the entropy of
#: the mean prediction is the score that carries the information (decisions.md 2026-11-23).
PRIMARY_SCORE_BY_BACKGROUND: dict[tuple[str, str], str] = {("uncertain", "mcdropout"): "entropy"}


def primary_score(method: str, background: str | None = None) -> str:
    """The primary score of ``method`` for runs trained with ``background.mode = background``."""
    return PRIMARY_SCORE_BY_BACKGROUND.get((background or "none", method), PRIMARY_SCORE[method])


def build(cfg: Config, classes: list[str], seed: int = 0):
    method = cfg.method
    K = len(classes)
    bg = cfg.get("background", {}) or {}
    mode = bg.get("mode", "none") or "none"
    if method == "softmax":
        from uaar.models.softmax import SoftmaxMethod

        return SoftmaxMethod(
            K,
            classes,
            dropout=cfg.model.dropout,
            seed=seed,
            background=mode,
            background_weight=float(bg.get("weight", 0.5)),
        )
    if method == "edl":
        from uaar.models.edl import EDLMethod

        e = cfg.get("model.edl", {}) or {}
        return EDLMethod(
            K,
            classes,
            dropout=cfg.model.dropout,
            evidence=e.get("evidence", "softplus"),
            loss=e.get("loss", "digamma"),
            kl_anneal_epochs=int(e.get("kl_anneal_epochs", 10)),
            kl_weight=float(e.get("kl_weight", 1.0)),
            seed=seed,
            background=mode,
            background_weight=float(bg.get("weight", 1.0)),
        )
    if method == "logreg":
        from uaar.models.logreg import LogRegMethod

        if mode != "none":
            raise ValueError("logreg has no background mode; it is a plain baseline only")
        return LogRegMethod(K, classes, seed=seed)
    if method == "mcdropout":
        raise ValueError(
            "mcdropout does not train: use evaluate.py --method mcdropout --from <softmax run>"
        )
    raise ValueError(f"unknown method {method!r}")


def load(run_dir: Path):
    """Load any saved method from its run directory by the ``method`` key in config.json."""
    import json

    method = json.loads((Path(run_dir) / "config.json").read_text())["method"]
    if method == "softmax":
        from uaar.models.softmax import SoftmaxMethod

        return SoftmaxMethod.load(Path(run_dir))
    if method == "edl":
        from uaar.models.edl import EDLMethod

        return EDLMethod.load(Path(run_dir))
    if method == "logreg":
        from uaar.models.logreg import LogRegMethod

        return LogRegMethod.load(Path(run_dir))
    if method == "mcdropout":
        from uaar.models.mcdropout import MCDropoutMethod

        return MCDropoutMethod.load(Path(run_dir))
    raise ValueError(method)
