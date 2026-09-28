"""The interventional-analogue decomposition of Eq. (1).

Ref: Eq. (1), p. 16::

    Psi(a, Q) = E[ integral mu_a(H, m) Q(dm | H) ]
    psi_dir   = Psi(1, g_{M|A=0}) - Psi(0, g_{M|A=0})
    psi_med   = Psi(1, g_{M|A=1}) - Psi(1, g_{M|A=0})
    psi       = psi_dir + psi_med          theta = psi_med / psi

Sec. 4.4, p. 17: the interventional analogue is used "instead of the natural effect"
because the natural decomposition would need an additional cross-world independence
assumption, and here the person who delivers the treatment is the person whose action
causes the effect. The mediator is binary (a hand-back either followed the window or did
not), so the integral over ``Q(dm | H)`` is a two-term sum.

Sec. 2.5, p. 4 adds the two diagnostics that accompany the fraction: mediator positivity
inside the certified region, and the pre-specified test that the fraction differs from
one half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimators.influence import InfluenceEstimate, difference, ratio
from ..utils.types import BoolArray, FloatArray

_EPS = 1e-9


@dataclass(frozen=True)
class PathEstimates:
    """The decomposed completion effect on the certified region."""

    total: InfluenceEstimate
    direct: InfluenceEstimate
    mediated: InfluenceEstimate

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            "total": self.total.as_dict(),
            "direct": self.direct.as_dict(),
            "mediated": self.mediated.as_dict(),
        }


def psi_functional(
    mu_low: FloatArray,
    mu_high: FloatArray,
    rule_high: FloatArray,
) -> FloatArray:
    """``Psi(a, g)`` for a binary mediator, row by row.

    ``mu_low`` is ``mu_a(H, 0)``, ``mu_high`` is ``mu_a(H, 1)`` and ``rule_high`` is
    ``g(M = 1 | H)`` under whichever rule ``Q`` the caller passes.
    """
    if not (mu_low.shape == mu_high.shape == rule_high.shape):
        raise ValueError("the functional needs three aligned vectors")
    return np.asarray(rule_high * mu_high + (1.0 - rule_high) * mu_low, dtype=np.float64)


def augmented_psi_terms(
    arm: int,
    alert: FloatArray,
    outcome: FloatArray,
    mediator: FloatArray,
    propensity_exposed: FloatArray,
    mu_low: FloatArray,
    mu_high: FloatArray,
    rule_high: FloatArray,
    observed_rule_high: FloatArray,
) -> FloatArray:
    """Row terms of the doubly-robust estimator of ``Psi(a, g)``.

    The plug-in part averages the arm regression over the mediator rule, and the
    augmentation reweights the observed residual ``Y - mu_a(H, M)`` by the likelihood
    ratio between the target rule and the rule that generated the data, under the exposure
    weight of the arm being evaluated. When the target rule is the observed one the ratio
    is one and the term collapses to the AIPW arm mean of Eq. (2), which is the algebraic
    identity the decomposition is verified against.

    ``arm`` selects which exposure weight applies: ``A / pi`` for ``a = 1`` and
    ``(1 - A) / (1 - pi)`` for ``a = 0``. Using one arm's weight for the other arm would
    make the plug-in inconsistent, so the arm is a required argument rather than a default.
    """
    if arm not in (0, 1):
        raise ValueError("the arm must be 0 or 1")
    plug_in = psi_functional(mu_low, mu_high, rule_high)
    # Both rules are held inside the open unit interval before the ratio is formed: the
    # likelihood ratio is only defined on the common support, and a rule that saturates at
    # one would otherwise divide zero by zero.
    safe_rule = np.clip(observed_rule_high, _EPS, 1.0 - _EPS)
    safe_target = np.clip(rule_high, _EPS, 1.0 - _EPS)
    safe_propensity = np.where(np.abs(propensity_exposed) < _EPS, _EPS, propensity_exposed)
    safe_control = np.where(np.abs(1.0 - propensity_exposed) < _EPS, _EPS, 1.0 - propensity_exposed)
    observed_mu = np.where(mediator > 0.5, mu_high, mu_low)
    taken = mediator > 0.5
    ratio_term = np.where(taken, safe_target, 1.0 - safe_target) / np.where(
        taken, safe_rule, 1.0 - safe_rule
    )
    exposure_weight = alert / safe_propensity if arm == 1 else (1.0 - alert) / safe_control
    augmentation = exposure_weight * ratio_term * (outcome - observed_mu)
    return np.asarray(plug_in + augmentation, dtype=np.float64)


def psi_estimate(terms: FloatArray, mask: BoolArray) -> InfluenceEstimate:
    """Mean of the row terms on a region, named for the record."""
    region = terms[mask]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name="psi", point=point, influence=np.asarray(region - point, dtype=np.float64)
    )


def decompose(
    arm_observed: InfluenceEstimate,
    arm_crossed: InfluenceEstimate,
    arm_control: InfluenceEstimate,
) -> PathEstimates:
    """Assemble ``psi``, ``psi_med`` and ``psi_dir`` from the three arms of Eq. (1).

    ``arm_observed`` is ``Psi(1, g_{M|A=1}) = E[Y(1, M(1))]``, ``arm_crossed`` is
    ``Psi(1, g_{M|A=0}) = E[Y(1, M(0))]`` and ``arm_control`` is
    ``Psi(0, g_{M|A=0}) = E[Y(0, M(0))]``.
    """
    total = difference("psi_total", arm_observed, arm_control)
    direct = difference("psi_direct", arm_crossed, arm_control)
    mediated = difference("psi_mediated", arm_observed, arm_crossed)
    return PathEstimates(total=total, direct=direct, mediated=mediated)


def mediated_fraction(paths: PathEstimates) -> InfluenceEstimate:
    """``theta = psi_med / psi`` with the influence-function ratio interval.

    Ref: Algorithm 3, step 9, p. 18.
    """
    return ratio(paths.mediated, paths.total, name="mediated_fraction")


def additivity_residual(paths: PathEstimates) -> float:
    """``psi - (psi_dir + psi_med)``, which is zero by construction and is asserted."""
    return float(paths.total.point - (paths.direct.point + paths.mediated.point))


def mediator_positivity(
    rule_high: FloatArray, mask: BoolArray, bound: float = 0.05
) -> dict[str, float]:
    """Share of certified rows whose mediator rule stays off both boundaries.

    Ref: Sec. 2.5, p. 4 - "Positivity of mediator model persisted in the certified region
    in which the hand-back property remained away from both boundaries of all windows,
    while mediator positivity did not continue outside the certified region".
    """
    if bound <= 0.0 or bound >= 0.5:
        raise ValueError("the positivity bound must lie in (0, 0.5)")
    region = rule_high[mask]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    interior = (region >= bound) & (region <= 1.0 - bound)
    return {
        "share_interior": float(np.mean(interior)),
        "min_rule": float(np.min(region)),
        "max_rule": float(np.max(region)),
        "bound": bound,
    }


def theta_null_test(estimate: InfluenceEstimate, null: float = 0.5) -> dict[str, float]:
    """Test that the mediated fraction is not one half, and lies inside the unit interval."""
    low, high = estimate.interval()
    return {
        "estimate": estimate.point,
        "null": null,
        "p_value": estimate.p_value(null),
        "ci_low": low,
        "ci_high": high,
        "inside_unit_interval": float(bool(low > 0.0 and high < 1.0)),
    }


__all__ = [
    "PathEstimates",
    "additivity_residual",
    "augmented_psi_terms",
    "decompose",
    "mediated_fraction",
    "mediator_positivity",
    "psi_estimate",
    "psi_functional",
    "theta_null_test",
]
