"""Bootstrap intervals.

Ref: Table 4 Panel E, p. 12 - "Bootstrap inference (1000 replicates)"; Sec. 4.9, p. 20 -
the bootstrap is the sensitivity analysis alongside the influence-function intervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

from ..utils.types import FloatArray, IntArray

ReplicateStatistic = Callable[[IntArray], float]


def bootstrap_replicates(
    statistic: ReplicateStatistic, n_rows: int, n_replicates: int, seed: int
) -> FloatArray:
    """Evaluate ``statistic`` on ``n_replicates`` resamples of the row index set."""
    if n_rows < 2:
        raise ValueError("the bootstrap needs at least two rows")
    if n_replicates < 10:
        raise ValueError("a usable bootstrap needs at least ten replicates")
    generator = np.random.default_rng(seed)
    draws = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        draws[replicate] = statistic(generator.integers(0, n_rows, size=n_rows, dtype=np.int64))
    return draws


def percentile_interval(draws: FloatArray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution."""
    alpha = 0.5 * (1.0 - level)
    low = float(np.quantile(draws, alpha))
    high = float(np.quantile(draws, 1.0 - alpha))
    return low, high


def normal_interval(draws: FloatArray, point: float, level: float = 0.95) -> tuple[float, float]:
    """Normal interval centred on the original estimate, with the bootstrap spread."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    spread = float(np.std(draws, ddof=1))
    return point - z * spread, point + z * spread


def bootstrap_summary(
    statistic: ReplicateStatistic,
    n_rows: int,
    point: float,
    n_replicates: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, float]:
    """Both interval flavours plus the spread, which is what the sensitivity row reports."""
    draws = bootstrap_replicates(statistic, n_rows, n_replicates, seed)
    low, high = percentile_interval(draws, level)
    normal_low, normal_high = normal_interval(draws, point, level)
    return {
        "point": float(point),
        "replicates": float(n_replicates),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "percentile_low": low,
        "percentile_high": high,
        "normal_low": normal_low,
        "normal_high": normal_high,
    }


__all__ = [
    "ReplicateStatistic",
    "bootstrap_replicates",
    "bootstrap_summary",
    "normal_interval",
    "percentile_interval",
]
