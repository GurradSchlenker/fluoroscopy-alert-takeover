"""Deterministic seeding.

Ref: none - release-internal reproducibility plumbing. The protocol is pre-specified
(Sec. 4.2, p. 14), so every stochastic stage runs from an explicitly recorded seed.
"""

from __future__ import annotations

import os
import random

import numpy as np

_DEFAULT_SEED = 20260101


def set_seed(seed: int = _DEFAULT_SEED) -> int:
    """Seed python, numpy and torch, and return the seed that was applied."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch  # noqa: PLC0415 - torch is optional for the causal-only path
    except ImportError:
        return seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def rng(seed: int, stream: str = "") -> np.random.Generator:
    """Return a generator keyed by ``seed`` and a stage name.

    Stages that must stay independent of each other's draw order get their own stream
    name, so adding a new stage cannot shift the numbers of an existing one.
    """
    if not stream:
        return np.random.default_rng(seed)
    digest = sum((index + 1) * ord(char) for index, char in enumerate(stream))
    return np.random.default_rng((seed + 7919 * digest) % 2**63)


__all__ = ["rng", "set_seed"]
