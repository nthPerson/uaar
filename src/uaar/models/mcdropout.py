"""M4: Monte Carlo Dropout (Gal & Ghahramani 2016). Owner: Sirena.

The weights are those of a trained M2 softmax network; nothing is trained here. At test time
the Dropout layers are left on, the network is run T times, and the spread of the T softmax
outputs is read as model (epistemic) uncertainty:

    p_bar     = (1/T) sum_t softmax(z_t)                    predictive mean
    entropy   = H[p_bar]                                    total uncertainty
    mi        = H[p_bar] - (1/T) sum_t H[softmax(z_t)]      mutual information (BALD)
    varratio  = 1 - (votes for the modal class) / T         variation ratio

Only the Dropout modules are switched to training mode; BatchNorm stays in eval mode so the
running statistics are used and not the statistics of the test batch (models.md §5).

Implemented from the design document; AI assistance recorded in docs/ai-use-log.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uaar.models import normalize
from uaar.models.base import Prediction, entropy
from uaar.models.softmax import SoftmaxMethod, SoftmaxNet


def enable_dropout_only(model: nn.Module) -> None:
    """Put ``model`` in eval mode, then switch every ``nn.Dropout`` back to training mode.

    BatchNorm (and everything else) stays in eval mode.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_scores(prob_sum: np.ndarray, ent_sum: np.ndarray, votes: np.ndarray, T: int) -> dict:
    """The three scores from per-window running sums over ``T`` stochastic passes.

    Args:
        prob_sum: ``(N, K)`` sum over passes of the softmax probabilities.
        ent_sum: ``(N,)`` sum over passes of each pass's entropy (nats).
        votes: ``(N, K)`` number of passes whose argmax was each class.
        T: number of passes.

    Returns:
        ``{"probs": (N, K), "entropy": (N,), "mi": (N,), "varratio": (N,)}``. ``mi`` is
        non-negative in exact arithmetic (Jensen); tiny negative rounding is clipped to 0.
    """
    probs = prob_sum / T
    h = entropy(probs)
    return {
        "probs": probs,
        "entropy": h,
        "mi": np.clip(h - ent_sum / T, 0.0, None),
        "varratio": 1.0 - votes.max(axis=1) / T,
    }


class MCDropoutMethod:
    """T stochastic forward passes through a trained softmax network.

    Args:
        net: the trained ``SoftmaxNet`` (weights are used as they are).
        classes: known-class names, index = integer label.
        stats: channel normalization statistics of the softmax run (``preprocess.json``).
        T: number of stochastic passes (30 by default; ablated over {5, 10, 30, 50}).
        seed: seeds the dropout masks, so ``predict`` is repeatable. The global PyTorch random
            state is restored afterwards.
        share_trunk: the backbone has Dropout only after the global pooling, so the
            convolutional part gives the same result in every pass. With ``True`` it is run
            once per batch and only the two-layer head is repeated T times; with ``False``
            the whole network is run T times. The outputs are the same (same masks); only the
            cost differs. Cost measurements should report both.
        source_run: run id of the softmax run the weights came from (recorded in config.json).
    """

    name = "mcdropout"
    score_names = ["entropy", "mi", "varratio"]

    def __init__(
        self,
        net: SoftmaxNet,
        classes: list[str],
        stats: dict,
        T: int = 30,
        seed: int = 0,
        share_trunk: bool = True,
        source_run: str | None = None,
    ):
        if T < 1:
            raise ValueError("T must be at least 1")
        self.model = net
        self.classes = classes
        self.stats = stats
        self.T = int(T)
        self.seed = seed
        self.share_trunk = share_trunk
        self.source_run = source_run

    @classmethod
    def from_softmax(
        cls, softmax: SoftmaxMethod, T: int = 30, seed: int = 0, **kwargs
    ) -> MCDropoutMethod:
        """Wrap a fitted or loaded ``SoftmaxMethod`` (shares its network object)."""
        assert softmax.stats is not None, "the softmax method must be fitted or loaded"
        return cls(softmax.model, softmax.classes, softmax.stats, T=T, seed=seed, **kwargs)

    @classmethod
    def from_softmax_run(
        cls, run_dir: Path, T: int = 30, seed: int = 0, **kwargs
    ) -> MCDropoutMethod:
        """Load the softmax run in ``run_dir`` and wrap it."""
        run_dir = Path(run_dir)
        kwargs.setdefault("source_run", run_dir.name)
        return cls.from_softmax(SoftmaxMethod.load(run_dir), T=T, seed=seed, **kwargs)

    def fit(self, Xtr, ytr, Xval, yval, cfg, run_dir: Path) -> None:
        raise RuntimeError(
            "mcdropout does not train: use evaluate.py --method mcdropout --from <softmax run>"
        )

    @torch.no_grad()
    def predict(self, X, cfg=None, batch_size: int = 1024) -> Prediction:
        """Predictive mean and the three scores for ``X`` float32 ``(N, 6, L)`` in g and deg/s."""
        device = next(self.model.parameters()).device
        was_training = self.model.training
        enable_dropout_only(self.model)
        N, K, T = len(X), len(self.classes), self.T
        prob_sum = np.zeros((N, K), dtype=np.float64)
        ent_sum = np.zeros(N, dtype=np.float64)
        votes = np.zeros((N, K), dtype=np.int64)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                for i in range(0, N, batch_size):
                    xb = torch.from_numpy(normalize.apply(X[i : i + batch_size], self.stats)).to(
                        device
                    )
                    rows = np.arange(len(xb))
                    pooled = None
                    if self.share_trunk:
                        pooled = self.model.pool(self.model.features(xb)).squeeze(-1)
                    for _ in range(T):
                        if pooled is not None:
                            z = self.model.head(self.model.fc(pooled))
                        else:
                            z = self.model(xb)
                        p = F.softmax(z, dim=1).cpu().numpy().astype(np.float64)
                        prob_sum[i : i + len(xb)] += p
                        ent_sum[i : i + len(xb)] += entropy(p)
                        votes[i + rows, p.argmax(axis=1)] += 1
        finally:
            self.model.train(was_training)
        s = mc_scores(prob_sum, ent_sum, votes, T)
        probs = s.pop("probs").astype(np.float32)
        return Prediction(probs=probs, label=probs.argmax(axis=1), scores=s)

    def save(self, run_dir: Path) -> None:
        """Write a self-contained run directory (weights, config.json, preprocess.json)."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), run_dir / "model.pt")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    **self.model.config(),
                    "classes": self.classes,
                    "seed": self.seed,
                    "method": self.name,
                    "T": self.T,
                    "share_trunk": self.share_trunk,
                    "source_run": self.source_run,
                },
                indent=2,
            )
        )
        normalize.save(
            self.stats,
            run_dir / "preprocess.json",
            window_samples=100,
            sample_rate_hz=50,
            channels=["ax", "ay", "az", "gx", "gy", "gz"],
        )

    @classmethod
    def load(cls, run_dir: Path) -> MCDropoutMethod:
        run_dir = Path(run_dir)
        conf = json.loads((run_dir / "config.json").read_text())
        net = SoftmaxNet(conf["n_classes"], dropout=conf["dropout"])
        net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        net.eval()
        return cls(
            net,
            conf["classes"],
            normalize.load(run_dir / "preprocess.json"),
            T=conf.get("T", 30),
            seed=conf.get("seed", 0),
            share_trunk=conf.get("share_trunk", True),
            source_run=conf.get("source_run"),
        )
