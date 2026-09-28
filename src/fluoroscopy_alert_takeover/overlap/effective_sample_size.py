"""Effective sample size of a gated region.

Ref: Algorithm 1, step 7, p. 16 - ``e_k = (sum_{O_k} pi)^2 / sum_{O_k} pi^2``, the Kish
effective sample size, tracked through the gate sweep as the trajectory ``{e_k}``.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import BoolArray, FloatArray


def kish_effective_sample_size(weights: FloatArray) -> float:
    """Kish effective sample size of a non-negative weight vector."""
    if weights.size == 0:
        raise ValueError("effective sample size needs at least one weight")
    if bool((weights < 0.0).any()):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    squared = float(np.sum(weights**2))
    if squared <= 0.0:
        return 0.0
    return total * total / squared


def effective_sample_size(propensity: FloatArray, mask: BoolArray) -> float:
    """Effective sample size of a gated region at the overlap weights ``pi (1 - pi)``.

    Algorithm 1 tracks the trajectory of the exposure propensity itself; the overlap
    weight is the same quantity rescaled by the constant ``1/4`` and is what the
    positivity probe reports.
    """
    selected = propensity[mask]
    if selected.size == 0:
        return 0.0
    return kish_effective_sample_size(np.asarray(selected, dtype=np.float64))


def normalised_effective_sample_size(propensity: FloatArray, mask: BoolArray) -> float:
    """Effective sample size divided by the size of the region."""
    count = float(np.count_nonzero(mask))
    if count == 0.0:
        return 0.0
    return effective_sample_size(propensity, mask) / count


__all__ = [
    "effective_sample_size",
    "kish_effective_sample_size",
    "normalised_effective_sample_size",
]
