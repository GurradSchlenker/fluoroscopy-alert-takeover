"""Between-site heterogeneity.

Ref: Table A2, p. 26 - "Heterogeneity: Cochran Q = 2.04 (df = 2), p = 0.36; I2 = 21.4%;
tau2 = 0.09"; Sec. 4.7, p. 19 - "at every research site, instead of using a unified pooled
statistic, sites show results, but they also provide a heterogeneity statistic".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..utils.types import FloatArray


@dataclass(frozen=True)
class Heterogeneity:
    """Cochran's Q with its p-value, the I-squared share and the DerSimonian-Laird tau-squared."""

    q: float
    df: int
    p_value: float
    i_squared: float
    tau_squared: float
    pooled: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cochran_q": self.q,
            "df": float(self.df),
            "p_value": self.p_value,
            "i_squared": self.i_squared,
            "tau_squared": self.tau_squared,
            "pooled": self.pooled,
        }


def cochran_q(
    estimates: FloatArray, standard_errors: FloatArray
) -> tuple[float, int, float, float]:
    """Cochran's Q, its degrees of freedom, its p-value and the fixed-effect mean."""
    values = np.asarray(estimates, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if values.shape != errors.shape:
        raise ValueError("estimates and standard errors must be aligned")
    if values.size < 2:
        raise ValueError("heterogeneity needs at least two estimates")
    if bool((errors <= 0.0).any()):
        raise ValueError("standard errors must be positive")
    weights = 1.0 / errors**2
    pooled = float(np.sum(weights * values) / np.sum(weights))
    q = float(np.sum(weights * (values - pooled) ** 2))
    df = int(values.size - 1)
    p_value = float(stats.chi2.sf(q, df))
    return q, df, p_value, pooled


def i_squared(q: float, df: int) -> float:
    """The I-squared share, as a percentage.

    Ref: the definition the manuscript's own table implies is not recoverable from its
    printed Q and df, so the conventional estimator is used here and the printed value is
    handled by the verifier as a manuscript-arithmetic finding.
    """
    if q <= 0.0:
        return 0.0
    if q <= df:
        return 0.0
    return float(100.0 * (q - df) / q)


def tau_squared(q: float, df: int, estimates: FloatArray, standard_errors: FloatArray) -> float:
    """DerSimonian-Laird between-study variance."""
    np.asarray(estimates, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    weights = 1.0 / errors**2
    denominator = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, (q - df) / denominator))


def heterogeneity_report(estimates: FloatArray, standard_errors: FloatArray) -> Heterogeneity:
    """Assemble the Table A2 footer."""
    q, df, p_value, pooled = cochran_q(estimates, standard_errors)
    return Heterogeneity(
        q=q,
        df=df,
        p_value=p_value,
        i_squared=i_squared(q, df),
        tau_squared=tau_squared(q, df, estimates, standard_errors),
        pooled=pooled,
    )


def pooled_estimate(estimates: FloatArray, sizes: FloatArray) -> float:
    """Size-weighted pooling of stratum point estimates.

    Table 3's two pooled rows are reproduced by weighting each stratum by its size rather
    than by its inverse variance, so the same convention is used here.
    """
    values = np.asarray(estimates, dtype=np.float64)
    counts = np.asarray(sizes, dtype=np.float64)
    if values.shape != counts.shape:
        raise ValueError("estimates and sizes must be aligned")
    if float(counts.sum()) <= 0.0:
        raise ValueError("sizes must carry positive mass")
    return float(np.sum(values * counts) / np.sum(counts))


__all__ = [
    "Heterogeneity",
    "cochran_q",
    "heterogeneity_report",
    "i_squared",
    "pooled_estimate",
    "tau_squared",
]
