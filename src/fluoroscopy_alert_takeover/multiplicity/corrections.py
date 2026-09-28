"""Multiplicity corrections.

Ref: Sec. 4.9, pp. 20-21 - "the establishment of multiplicity takes place. While one
category includes main comparisons that have been corrected for family-wise error through
the step-down technique, the other category contains varying types of comparisons like
the per-site, per-stratum and per-group comparisons are controlled at a false-discovery
rate of 0.05."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import BoolArray, FloatArray


@dataclass(frozen=True)
class CorrectedPValues:
    """Adjusted p-values in the order the raw values were supplied."""

    family: str
    raw: FloatArray
    adjusted: FloatArray
    rejected: BoolArray
    alpha: float

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "raw": [float(value) for value in self.raw],
            "adjusted": [float(value) for value in self.adjusted],
            "rejected": [bool(value) for value in self.rejected],
            "alpha": self.alpha,
        }


def holm_step_down(p_values: FloatArray, alpha: float = 0.05) -> CorrectedPValues:
    """Holm step-down adjustment of the primary family."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("the step-down procedure needs a flat non-empty family")
    order = np.argsort(values, kind="stable")
    count = values.size
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = (count - rank) * float(values[position])
        running = max(running, min(1.0, candidate))
        adjusted[position] = running
    rejected = adjusted <= alpha
    return CorrectedPValues(
        family="holm_step_down", raw=values, adjusted=adjusted, rejected=rejected, alpha=alpha
    )


def benjamini_hochberg(p_values: FloatArray, alpha: float = 0.05) -> CorrectedPValues:
    """Benjamini-Hochberg step-up adjustment of the secondary family."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("the step-up procedure needs a flat non-empty family")
    order = np.argsort(values, kind="stable")
    count = values.size
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for rank in range(count - 1, -1, -1):
        candidate = float(values[order[rank]]) * count / (rank + 1)
        running = min(running, 1.0, candidate)
        adjusted[order[rank]] = running
    rejected = adjusted <= alpha
    return CorrectedPValues(
        family="benjamini_hochberg", raw=values, adjusted=adjusted, rejected=rejected, alpha=alpha
    )


def two_family_report(
    primary: FloatArray, secondary: FloatArray, alpha: float = 0.05, fdr: float = 0.05
) -> dict[str, object]:
    """Apply the two declared families of Sec. 4.9 and return both records."""
    return {
        "primary": holm_step_down(primary, alpha).as_dict(),
        "secondary": benjamini_hochberg(secondary, fdr).as_dict(),
    }


__all__ = ["CorrectedPValues", "benjamini_hochberg", "holm_step_down", "two_family_report"]
