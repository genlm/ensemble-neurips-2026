import random
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, n_bootstraps=10000, ci=95, seed=42):
    """Mean and percentile-bootstrap CI of the mean (the paper's per-seed CI)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.array(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstraps)]
    )
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lower), float(upper)


def safe_decode(b: bytes) -> str:
    """Decode one token's bytes; undecodable (partial UTF-8) tokens become ''.

    This is how the paper's token-level runners turned particles into text, so it is
    kept as-is for comparability.
    """
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def resolve_op(op):
    """Map the paper's op names onto ``genlm.control.Ensemble`` ops.

    ``min``, ``max``, ``sum``, ``prod``, ``harmonic`` pass through; ``p2`` / ``p-2`` /
    ``pm2.5`` denote the power mean with that exponent (``pm`` = negative).
    """
    if isinstance(op, str):
        m = re.fullmatch(r"p(m?)(-?[0-9.]+)", op)
        if m:
            return -float(m.group(2)) if m.group(1) else float(m.group(2))
    return op


def short(model_name: str) -> str:
    return model_name.split("/")[-1]
