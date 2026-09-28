"""Dice coefficient and the soft surrogate used to train the spatial predicate.

Ref: Sec. 2.2, p. 3 - "The Dice coefficient of 0.809 was calculated using a spatial
predicate applied to the clinical fluoroscopic stream"; Table A2, p. 26 - the per-site
Dice values.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-6


def dice_coefficient(prediction: Tensor, reference: Tensor, threshold: float = 0.5) -> float:
    """Hard Dice between a mask logit map and a binary reference.

    Ref: Sec. 2.2, p. 3. Both inputs are ``(N, 1, H, W)``; the coefficient is computed on
    the flattened batch, which is how a whole held-out cine stream is scored.
    """
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must share a shape")
    predicted = (torch.sigmoid(prediction) >= threshold).to(reference.dtype)
    target = (reference > 0.5).to(reference.dtype)
    intersection = float((predicted * target).sum().item())
    denominator = float(predicted.sum().item() + target.sum().item())
    if denominator == 0.0:
        return 1.0
    return 2.0 * intersection / denominator


def soft_dice_loss(prediction: Tensor, reference: Tensor) -> Tensor:
    """Differentiable Dice surrogate, averaged over the batch."""
    probability = torch.sigmoid(prediction)
    target = (reference > 0.5).to(probability.dtype)
    dims = tuple(range(1, probability.dim()))
    intersection = (probability * target).sum(dim=dims)
    cardinality = probability.sum(dim=dims) + target.sum(dim=dims)
    score = (2.0 * intersection + _EPS) / (cardinality + _EPS)
    return 1.0 - score.mean()


def per_site_dice(
    prediction: Tensor, reference: Tensor, site_index: Tensor, n_sites: int, threshold: float = 0.5
) -> list[float]:
    """Dice computed within each site, for the cross-site consistency check."""
    values: list[float] = []
    for site in range(n_sites):
        mask = site_index == site
        if not bool(mask.any()):
            values.append(float("nan"))
            continue
        values.append(dice_coefficient(prediction[mask], reference[mask], threshold))
    return values


__all__ = ["dice_coefficient", "per_site_dice", "soft_dice_loss"]
