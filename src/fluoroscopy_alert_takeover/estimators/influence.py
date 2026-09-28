"""Influence-function intervals.

Ref: Algorithm 3, step 7, p. 18 ("assemble psi by Eq. (2) and its interval from the
influence function"); Sec. 4.9, p. 20 - "The influence function of the one-step estimator
yields a large number of confidence intervals".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..utils.types import FloatArray

_EPS = 1e-12


@dataclass(frozen=True)
class InfluenceEstimate:
    """A point estimate carried by the influence function of every row it was built from."""

    name: str
    point: float
    influence: FloatArray

    @property
    def n(self) -> int:
        return int(self.influence.shape[0])

    @property
    def standard_error(self) -> float:
        if self.n < 2:
            return float("nan")
        return float(np.std(self.influence, ddof=1) / np.sqrt(self.n))

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Normal interval from the influence-function standard error."""
        se = self.standard_error
        if not np.isfinite(se) or se == 0.0:
            return self.point, self.point
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return self.point - z * se, self.point + z * se

    def p_value(self, null: float = 0.0) -> float:
        se = self.standard_error
        if not np.isfinite(se) or se == 0.0:
            return float("nan")
        z = (self.point - null) / se
        return float(2.0 * stats.norm.sf(abs(z)))

    def covers(self, value: float = 0.0, level: float = 0.95) -> bool:
        low, high = self.interval(level)
        return bool(low <= value <= high)

    def as_dict(self, level: float = 0.95) -> dict[str, float]:
        low, high = self.interval(level)
        return {
            "point": self.point,
            "se": self.standard_error,
            "ci_low": low,
            "ci_high": high,
            "p_value": self.p_value(),
            "n": float(self.n),
        }


def centred_influence(values: FloatArray, point: float) -> FloatArray:
    """Centre a pseudo-outcome vector on its own mean."""
    return np.asarray(values - point, dtype=np.float64)


def interval_from_influence(
    point: float, influence: FloatArray, level: float = 0.95
) -> tuple[float, float]:
    """Convenience wrapper returning only the interval of a point and its influence."""
    return InfluenceEstimate(name="contrast", point=point, influence=influence).interval(level)


def difference(name: str, left: InfluenceEstimate, right: InfluenceEstimate) -> InfluenceEstimate:
    """Influence-function difference of two estimates computed on the same rows."""
    if left.influence.shape[0] != right.influence.shape[0]:
        raise ValueError("both estimates must be computed on the same rows")
    return InfluenceEstimate(
        name=name,
        point=left.point - right.point,
        influence=np.asarray(left.influence - right.influence, dtype=np.float64),
    )


def ratio(
    numerator: InfluenceEstimate, denominator: InfluenceEstimate, name: str = "ratio"
) -> InfluenceEstimate:
    """Ratio of two estimates with the delta-method influence function.

    Ref: Algorithm 3, step 9, p. 18 - ``theta = psi_med / psi`` with "an influence-function
    interval for the ratio".
    """
    if numerator.influence.shape[0] != denominator.influence.shape[0]:
        raise ValueError("both estimates must be computed on the same rows")
    if abs(denominator.point) <= _EPS:
        raise ZeroDivisionError("the denominator contrast is zero; the ratio is undefined")
    point = numerator.point / denominator.point
    influence = (numerator.influence - point * denominator.influence) / denominator.point
    return InfluenceEstimate(
        name=name, point=float(point), influence=np.asarray(influence, dtype=np.float64)
    )


def linear_contrast(name: str, parts: list[tuple[float, InfluenceEstimate]]) -> InfluenceEstimate:
    """Weighted sum of estimates, used by the pooled strata and the gradient test."""
    if not parts:
        raise ValueError("a linear contrast needs at least one term")
    size = parts[0][1].influence.shape[0]
    for _, estimate in parts:
        if estimate.influence.shape[0] != size:
            raise ValueError("all terms must be computed on the same rows")
    point = float(sum(weight * estimate.point for weight, estimate in parts))
    influence = np.zeros(size, dtype=np.float64)
    for weight, estimate in parts:
        influence = influence + weight * estimate.influence
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(influence, dtype=np.float64)
    )


__all__ = [
    "InfluenceEstimate",
    "centred_influence",
    "difference",
    "interval_from_influence",
    "linear_contrast",
    "ratio",
]
