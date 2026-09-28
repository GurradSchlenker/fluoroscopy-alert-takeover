"""The bias-factor curve and its null crossing.

Ref: Algorithm 2, steps 7-9, p. 16 - "for gamma in Gamma do recompute the primary contrast
under bias factor gamma and store the curve"; Table 4 Panel C, p. 11 - the row "Bias factor
Gamma at null crossing".

The curve reads the contrast under an unmeasured confounder that multiplies the exposed
risk by ``gamma``, so the factor at which the contrast reaches zero is the strength the
confounder would need to explain the result away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import FloatArray

DEFAULT_GAMMA_GRID: tuple[float, ...] = tuple(np.round(np.arange(1.0, 2.01, 0.01), 4))


@dataclass(frozen=True)
class BiasFactorCurve:
    """Contrast as a function of the bias factor."""

    gamma: FloatArray
    contrast: FloatArray
    crossing: float

    def as_dict(self) -> dict[str, object]:
        return {
            "gamma": [float(value) for value in self.gamma],
            "contrast": [float(value) for value in self.contrast],
            "null_crossing": self.crossing,
        }


def contrast_under_bias(risk_difference: float, unexposed_risk: float, gamma: float) -> float:
    """Risk difference after dividing the exposed risk by ``gamma``."""
    if gamma < 1.0:
        raise ValueError("the bias factor is at least one")
    if not 0.0 < unexposed_risk < 1.0:
        raise ValueError("the unexposed risk must lie strictly inside (0, 1)")
    exposed = (unexposed_risk + risk_difference) / gamma
    return float(exposed - unexposed_risk)


def bias_factor_curve(
    risk_difference: float,
    unexposed_risk: float,
    grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
) -> BiasFactorCurve:
    """Evaluate the curve and locate its first non-positive value."""
    values = np.asarray(
        [contrast_under_bias(risk_difference, unexposed_risk, g) for g in grid], dtype=np.float64
    )
    crossing = next(
        (float(g) for g, value in zip(grid, values, strict=True) if value <= 0.0),
        float("nan"),
    )
    return BiasFactorCurve(
        gamma=np.asarray(grid, dtype=np.float64), contrast=values, crossing=crossing
    )


def null_crossing(risk_difference: float, unexposed_risk: float) -> float:
    """Bias factor at which the contrast reaches the null, in closed form."""
    if risk_difference <= 0.0:
        return 1.0
    return float((unexposed_risk + risk_difference) / unexposed_risk)


__all__ = [
    "DEFAULT_GAMMA_GRID",
    "BiasFactorCurve",
    "bias_factor_curve",
    "contrast_under_bias",
    "null_crossing",
]
