"""The augmented inverse-probability-weighted one-step estimator of Eq. (2).

Ref: Eq. (2), p. 17, transcribed term by term::

    psi_AIPW = 1/|O| * sum_{(i,w) in O} [
        A_{i,w} * (Y_i - mu_1(H_{i,w})) / pi(H_{i,w})
      - (1 - A_{i,w}) * (Y_i - mu_0(H_{i,w})) / (1 - pi(H_{i,w}))
      + mu_1(H_{i,w}) - mu_0(H_{i,w}) ]

The average is taken over the certified region alone, because Sec. 4.4, p. 17 states that
the certified area "should be considered a modification of the estimand and not simply a
purification procedure"; the influence function is therefore the centred pseudo-outcome
within that region.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import BoolArray, FloatArray
from .influence import InfluenceEstimate

_EPS = 1e-9


def eq2_terms(
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
) -> FloatArray:
    """The bracketed term of Eq. (2) for every row.

    Denominators are guarded only against exact zero: the propensity has already been
    clipped by Algorithm 3, step 6, so a guard here would silently undo the trimming.
    """
    if not (
        alert.shape == outcome.shape == propensity.shape == mu_treated.shape == mu_control.shape
    ):
        raise ValueError("all inputs of Eq. (2) must be aligned")
    safe_exposed = np.where(np.abs(propensity) < _EPS, _EPS, propensity)
    safe_control = np.where(np.abs(1.0 - propensity) < _EPS, _EPS, 1.0 - propensity)
    exposed = alert * (outcome - mu_treated) / safe_exposed
    control = (1.0 - alert) * (outcome - mu_control) / safe_control
    return np.asarray(exposed - control + mu_treated - mu_control, dtype=np.float64)


def aipw_contrast(
    name: str,
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray | None = None,
) -> InfluenceEstimate:
    """Eq. (2) evaluated on a region, with its influence function."""
    terms = eq2_terms(alert, outcome, propensity, mu_treated, mu_control)
    selected = np.ones(terms.shape[0], dtype=np.bool_) if mask is None else mask
    region = terms[selected]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name=name,
        point=point,
        influence=np.asarray(region - point, dtype=np.float64),
    )


def aipw_arm_means(
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray | None = None,
) -> tuple[float, float]:
    """The two arm means of Eq. (2) separately, for the decomposition of Eq. (1)."""
    safe_exposed = np.where(np.abs(propensity) < _EPS, _EPS, propensity)
    safe_control = np.where(np.abs(1.0 - propensity) < _EPS, _EPS, 1.0 - propensity)
    mean_treated_rows = alert * (outcome - mu_treated) / safe_exposed + mu_treated
    mean_control_rows = (1.0 - alert) * (outcome - mu_control) / safe_control + mu_control
    selected = np.ones(alert.shape[0], dtype=np.bool_) if mask is None else mask
    return (
        float(np.mean(mean_treated_rows[selected])),
        float(np.mean(mean_control_rows[selected])),
    )


__all__ = ["aipw_arm_means", "aipw_contrast", "eq2_terms"]
