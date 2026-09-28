"""The hand-back window-length attenuation prediction.

Ref: Table 4 Panel B, p. 11 - the mediated fraction falls from 0.73 at the pre-specified
window to 0.61, 0.44 and 0.29 as the window is doubled, tripled and quadrupled; Sec. 2.5,
p. 4 - "the mediated fraction decreases over time".

The prediction is what separates a response channel from an accumulation of correlations:
if the mediator is a response to the alert, extending the window past the decision point
must dilute its share of the effect. Only the mediator moves; the exposure and the outcome
stay at their pre-specified values.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..estimands.interventional import mediated_fraction
from ..estimators.crossfit import clip_propensity
from ..estimators.decomposition import cross_fitted_decomposition, cross_fitted_mediation_nuisance
from ..exposure.model import fit_balance_first_exposure
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class AttenuationPoint:
    """The mediated fraction at one window multiplier."""

    multiplier: float
    fraction: float
    interval_low: float
    interval_high: float
    mediator_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "multiplier": self.multiplier,
            "mediated_fraction": self.fraction,
            "ci_low": self.interval_low,
            "ci_high": self.interval_high,
            "mediator_rate": self.mediator_rate,
        }


def window_attenuation(
    table: PersonWindowTable,
    folds: IntArray,
    outcome_name: str,
    learner: str,
    mask: BoolArray,
    clip_gamma: float,
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    propensity: FloatArray | None = None,
) -> list[AttenuationPoint]:
    """Mediated fraction at each window multiplier.

    The exposure model is held fixed across the sweep, because extending the observation
    window after the alert cannot change the covariates that precede it; only the mediator
    rule is refitted, on the widened mediator.
    """
    if propensity is None:
        exposure = fit_balance_first_exposure(table, folds)
        propensity = clip_propensity(exposure.propensity, clip_gamma).propensity
    if sorted(multipliers) != list(multipliers):
        raise ValueError("the multiplier grid must be increasing")

    points: list[AttenuationPoint] = []
    for multiplier in multipliers:
        widened = table.with_mediator_window(multiplier)
        nuisance = cross_fitted_mediation_nuisance(
            widened, propensity, folds, outcome_name, learner
        )
        paths, _ = cross_fitted_decomposition(widened, nuisance, mask, outcome_name)
        fraction = mediated_fraction(paths)
        low, high = fraction.interval()
        points.append(
            AttenuationPoint(
                multiplier=float(multiplier),
                fraction=fraction.point,
                interval_low=low,
                interval_high=high,
                mediator_rate=float(np.mean(widened.hand_back[mask])),
            )
        )
    return points


def monotone_attenuation(points: list[AttenuationPoint]) -> dict[str, object]:
    """Whether the sweep falls monotonically, which is the prediction of Sec. 2.5."""
    fractions = [point.fraction for point in points]
    decreasing = all(later <= earlier for earlier, later in itertools.pairwise(fractions))
    return {
        "multipliers": [point.multiplier for point in points],
        "fractions": fractions,
        "monotone_decreasing": decreasing,
        "total_drop": fractions[0] - fractions[-1] if fractions else float("nan"),
    }


__all__ = ["AttenuationPoint", "monotone_attenuation", "window_attenuation"]
