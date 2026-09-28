"""Cluster-robust standard errors.

Ref: Sec. 4.9, p. 20 - "The influence function of the one-step estimator yields a large
number of confidence intervals, as well as cluster-robust standard errors at both the site
level and the operator level"; Table 4 Panel E, p. 12 - the operator-level row.
"""

from __future__ import annotations

import numpy as np

from ..estimators.influence import InfluenceEstimate
from ..protocol.schema import PersonWindowTable
from ..utils.types import FloatArray, IntArray

CLUSTER_LEVELS: tuple[str, ...] = ("site", "operator")


def cluster_labels(table: PersonWindowTable, level: str) -> IntArray:
    """Integer cluster labels at the requested level."""
    if level == "operator":
        return table.operator_id.astype(np.int64)
    if level == "site":
        _, labels = np.unique(table.site, return_inverse=True)
        return np.asarray(labels, dtype=np.int64)
    raise KeyError(f"unknown cluster level: {level}")


def cluster_robust_standard_error(influence: FloatArray, clusters: IntArray) -> float:
    """Sandwich standard error with a finite-cluster correction.

    ``se^2 = G / ((G - 1) n^2) * sum_g (sum_{i in g} D_i)^2``, the usual clustered
    variance of a mean of influence contributions.
    """
    if influence.shape[0] != clusters.shape[0]:
        raise ValueError("influence and clusters must be aligned")
    unique = np.unique(clusters)
    guard = unique.shape[0]
    if guard < 2:
        return float("nan")
    totals = np.asarray(
        [float(np.sum(influence[clusters == value])) for value in unique], dtype=np.float64
    )
    size = float(influence.shape[0])
    variance = (guard / (guard - 1.0)) * float(np.sum(totals**2)) / (size * size)
    return float(np.sqrt(variance))


def cluster_robust_estimate(
    estimate: InfluenceEstimate, clusters: IntArray, level: str
) -> InfluenceEstimate:
    """Re-express an estimate with a cluster-robust standard error.

    The influence vector is scaled so that :meth:`InfluenceEstimate.standard_error`
    returns the clustered value, leaving the point estimate untouched.
    """
    se = cluster_robust_standard_error(estimate.influence, clusters)
    if not np.isfinite(se):
        return estimate
    target = se * np.sqrt(estimate.n)
    centred = estimate.influence - float(np.mean(estimate.influence))
    spread = float(np.std(centred, ddof=1))
    if spread <= 0.0:
        return estimate
    scaled = centred * (target / spread)
    return InfluenceEstimate(
        name=f"{estimate.name}:cluster_{level}", point=estimate.point, influence=scaled
    )


__all__ = [
    "CLUSTER_LEVELS",
    "cluster_labels",
    "cluster_robust_estimate",
    "cluster_robust_standard_error",
]
