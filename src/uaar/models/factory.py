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


def build(cfg: Config, classes: list[str], seed: int = 0):
    method = cfg.method
    K = len(classes)
    if method == "softmax":
        from uaar.models.softmax import SoftmaxMethod

        return SoftmaxMethod(K, classes, dropout=cfg.model.dropout, seed=seed)
    if method == "edl":
        from uaar.models.edl import EDLMethod

        e = cfg.get("model.edl", {}) or {}
        return EDLMethod(
            K,
            classes,
            dropout=cfg.model.dropout,
            evidence=e.get("evidence", "softplus"),
            loss=e.get("loss", "mse"),
            kl_anneal_epochs=int(e.get("kl_anneal_epochs", 10)),
            seed=seed,
        )
    if method == "logreg":
        from uaar.models.logreg import LogRegMethod

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
