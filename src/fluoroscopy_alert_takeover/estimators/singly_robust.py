"""The two singly-robust estimators of the Table 2 ablation.

Ref: Table 2, p. 8 - "the two singly-robust estimators disagree by 2.1 pp, while the
doubly-robust estimate moves by less than 0.2 pp when either nuisance family is replaced
by a machine-learning alternative". The rows ``IPW only`` and ``Outcome regression only``
are the two singly-robust members.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import BoolArray, FloatArray
from .influence import InfluenceEstimate

_EPS = 1e-9


def ipw_contrast(
    name: str,
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mask: BoolArray | None = None,
) -> InfluenceEstimate:
    """Inverse-probability weighting with no outcome regression."""
    safe_exposed = np.where(np.abs(propensity) < _EPS, _EPS, propensity)
    safe_control = np.where(np.abs(1.0 - propensity) < _EPS, _EPS, 1.0 - propensity)
    terms = alert * outcome / safe_exposed - (1.0 - alert) * outcome / safe_control
    selected = np.ones(terms.shape[0], dtype=np.bool_) if mask is None else mask
    region = terms[selected]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(region - point, dtype=np.float64)
    )


def outcome_regression_contrast(
    name: str,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray | None = None,
) -> InfluenceEstimate:
    """The g-computation contrast with no exposure weighting.

    The influence function here is the plug-in spread of the regression contrast; it does
    not carry the nuisance-estimation correction, which is the sense in which this row is
    singly robust.
    """
    terms = mu_treated - mu_control
    selected = np.ones(terms.shape[0], dtype=np.bool_) if mask is None else mask
    region = terms[selected]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(region - point, dtype=np.float64)
    )


__all__ = ["ipw_contrast", "outcome_regression_contrast"]
