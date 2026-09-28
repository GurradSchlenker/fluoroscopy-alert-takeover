"""Alternative analysis choices of Table 4 Panel E.

Ref: Table 4 Panel E, p. 12 - the stochastic-intervention estimand, cluster-robust
inference at the operator level, bootstrap inference with 1,000 replicates, and the
leave-one-site-out range; Sec. 4.7, p. 19 - "one site will not enter into each fitting
iteration and the frozen models can be validated as an external cohort".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..estimators.influence import InfluenceEstimate
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray
from .bootstrap import bootstrap_summary
from .robust import cluster_labels, cluster_robust_estimate

ContrastPair = Callable[[PersonWindowTable], tuple[InfluenceEstimate, InfluenceEstimate]]

_EPS = 1e-6


def stochastic_intervention_contrast(
    name: str,
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray,
    shift: float,
) -> InfluenceEstimate:
    """Effect of shifting the exposure mechanism by ``shift`` on the propensity scale.

    The intervention replaces the alert probability ``pi(H)`` with
    ``clip(pi(H) + shift)`` while the covariate history is held fixed, and the contrast is
    taken against the observed mechanism. The term

        (mu_1 - mu_0)(q_delta - pi) + (q_delta(A|H)/pi(A|H) - 1)(Y - mu_A(H))

    is the density-ratio form of that contrast, so it stays consistent under either a
    correct outcome regression or a correct exposure model, like the primary estimator.
    """
    target = np.clip(propensity + shift, _EPS, 1.0 - _EPS)
    observed = np.clip(propensity, _EPS, 1.0 - _EPS)
    mu_observed = np.where(alert > 0.5, mu_treated, mu_control)
    target_density = np.where(alert > 0.5, target, 1.0 - target)
    observed_density = np.where(alert > 0.5, observed, 1.0 - observed)
    terms = (mu_treated - mu_control) * (target - observed) + (
        target_density / observed_density - 1.0
    ) * (outcome - mu_observed)
    region = terms[mask]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(region - point, dtype=np.float64)
    )


def solve_shift_for_marginal_rate(
    propensity: FloatArray, alert: FloatArray, lower: float = -0.5, upper: float = 0.5
) -> float:
    """Shift whose marginal alert rate matches the observed share.

    Solving for the shift keeps the stochastic intervention comparable in burden to the
    deployed policy; without it the contrast would mix a behavioural effect with a change
    in how often the alert fires at all.
    """
    target = float(np.mean(alert))
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        rate = float(np.mean(np.clip(propensity + midpoint, 0.0, 1.0)))
        if rate > target:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def cluster_robust_rows(
    estimate: InfluenceEstimate,
    table: PersonWindowTable,
    mask: BoolArray,
    levels: tuple[str, ...] = ("site", "operator"),
) -> dict[str, dict[str, object]]:
    """The cluster-robust rows of Table 4 Panel E.

    The influence function lives on the certified region's rows, so the cluster labels are
    restricted to the same region before the sandwich is formed.
    """
    rows: dict[str, dict[str, object]] = {}
    for level in levels:
        clusters = cluster_labels(table, level)[mask]
        if np.unique(clusters).shape[0] < 2:
            summary: dict[str, object] = {
                "point": estimate.point,
                "note": "fewer than two clusters in the region",
            }
            rows[level] = summary
            continue
        payload: dict[str, object] = {"point": estimate.point}
        payload.update(cluster_robust_estimate(estimate, clusters, level).as_dict())
        rows[level] = payload
    return rows


def bootstrap_row(
    estimate: InfluenceEstimate,
    n_replicates: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap the certified region's pseudo-outcome mean."""
    influence = estimate.influence

    def statistic(indices: np.ndarray) -> float:
        return float(estimate.point + np.mean(influence[indices]))

    return bootstrap_summary(
        statistic, influence.shape[0], estimate.point, n_replicates, seed, level
    )


def leave_one_site_out(
    table: PersonWindowTable, contrast: ContrastPair, outcome_index: int = 0
) -> dict[str, object]:
    """Refit the whole analysis with each site held out in turn.

    Ref: Sec. 4.7, p. 19. The held-out site takes no part in the fitting of that
    configuration, which is the external-cohort reading of the procedure. The contrast
    callable returns both reported outcomes, so the caller names which one the range is
    taken over.
    """
    if outcome_index not in (0, 1):
        raise ValueError("the outcome index selects one of the two reported contrasts")
    sites = sorted({str(value) for value in table.site})
    if len(sites) < 2:
        raise ValueError("leave-one-site-out needs at least two sites")
    points: dict[str, float] = {}
    for site in sites:
        subset = table.subset(np.asarray(table.site != site, dtype=np.bool_))
        points[site] = contrast(subset)[outcome_index].point
    values = np.asarray(list(points.values()), dtype=np.float64)
    return {
        "held_out": points,
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "crossed_null": bool((values <= 0.0).any()),
    }


__all__ = [
    "ContrastPair",
    "bootstrap_row",
    "cluster_robust_rows",
    "leave_one_site_out",
    "solve_shift_for_marginal_rate",
    "stochastic_intervention_contrast",
]
