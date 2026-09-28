"""Targeted maximum likelihood, the alternative estimator of Table 2.

Ref: Sec. 4.4, p. 17 - "the maximum-likelihood version of targeted learning used as an
alternative"; Table 2, p. 8 - the row ``TMLE, cross-fitted``.

The fluctuation is the standard logistic one for a bounded outcome: the initial arm
regressions move along the clever covariate ``A/pi - (1-A)/(1-pi)`` until the efficient
score is solved. The one-parameter fit is solved by bisection rather than by a general
GLM, because the score equation is strictly monotone in the fluctuation.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import BoolArray, FloatArray
from .influence import InfluenceEstimate

_EPS = 1e-6
_TOL = 1e-10


def _logit(probability: FloatArray) -> FloatArray:
    clipped = np.clip(probability, _EPS, 1.0 - _EPS)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def _expit(value: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-value)), dtype=np.float64)


def clever_covariate(alert: FloatArray, propensity: FloatArray) -> FloatArray:
    """``H(A, H) = A/pi - (1-A)/(1-pi)``."""
    safe_exposed = np.where(np.abs(propensity) < _EPS, _EPS, propensity)
    safe_control = np.where(np.abs(1.0 - propensity) < _EPS, _EPS, 1.0 - propensity)
    return np.asarray(alert / safe_exposed - (1.0 - alert) / safe_control, dtype=np.float64)


def fit_fluctuation(
    initial: FloatArray, covariate: FloatArray, outcome: FloatArray, bracket: float = 10.0
) -> float:
    """Solve ``sum covariate * (outcome - expit(offset + eps * covariate)) = 0``.

    The score is strictly decreasing in ``eps`` because its derivative is
    ``-sum covariate^2 * p(1-p)``, so bisection on a fixed bracket is exact here.
    """
    offset = _logit(initial)

    def score(epsilon: float) -> float:
        residual = outcome - _expit(offset + epsilon * covariate)
        return float(np.sum(covariate * residual))

    low, high = -bracket, bracket
    if score(low) < 0.0 or score(high) > 0.0:
        return 0.0 if abs(score(0.0)) < _TOL else float(np.sign(score(0.0)) * bracket)
    for _ in range(200):
        midpoint = 0.5 * (low + high)
        if score(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
        if high - low < _TOL:
            break
    return 0.5 * (low + high)


def tmle_contrast(
    name: str,
    alert: FloatArray,
    outcome: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray | None = None,
) -> InfluenceEstimate:
    """Targeted estimate of the risk difference on the certified region."""
    selected = np.ones(alert.shape[0], dtype=np.bool_) if mask is None else mask
    covariate = clever_covariate(alert, propensity)
    initial = np.where(alert > 0.5, mu_treated, mu_control)
    epsilon = fit_fluctuation(initial[selected], covariate[selected], outcome[selected])

    targeted_exposed = _expit(
        _logit(mu_treated) + epsilon / np.where(propensity < _EPS, _EPS, propensity)
    )
    targeted_control = _expit(
        _logit(mu_control) - epsilon / np.where((1.0 - propensity) < _EPS, _EPS, 1.0 - propensity)
    )
    targeted = np.where(alert > 0.5, targeted_exposed, targeted_control)
    contrast = targeted_exposed - targeted_control
    point = float(np.mean(contrast[selected]))
    influence = (covariate * (outcome - targeted) + contrast - point)[selected]
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(influence, dtype=np.float64)
    )


__all__ = ["clever_covariate", "fit_fluctuation", "tmle_contrast"]
