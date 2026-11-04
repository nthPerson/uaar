"""Small shared helpers: git SHA for provenance, seeding, timing."""

from __future__ import annotations

import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    """The directory containing ``pyproject.toml`` (walking up from this file)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def git_sha(short: bool = True) -> str:
    """Current commit of the repository, ``"unknown"`` outside git; ``-dirty`` when modified."""
    try:
        root = repo_root()
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seed_everything(seed: int) -> None:
    """Seed ``random``, NumPy and (when installed) PyTorch on CPU and CUDA, deterministically."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (libraries still read the legacy global RNG)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextmanager
def timer(label: str):
    """``with timer("loading"):`` prints the elapsed seconds when the block ends."""
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.1f} s")
