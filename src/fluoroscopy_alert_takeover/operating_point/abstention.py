"""The abstention band, its operating point and the constrained comparison.

Ref: Algorithm 5, step 6, p. 20 - "select tau_op as the maximiser of the pre-specified
precision and interrupt-burden criterion over T; freeze tau_op and propagate it to every
downstream analysis"; Table 4 Panel D, p. 12 - the band policy is "required to show parity
on primary utility and to win only on the constraint axis".

The criterion itself is pre-specified by the investigators and not printed; the release
exposes the two knobs it must consume - a burden ceiling and a utility floor - and labels
their defaults as engineering values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.discrimination import auroc, metric_interval
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray
from .benefit_curve import BenefitCurve

DEFAULT_BURDEN_CEILING = 0.60
DEFAULT_UTILITY_FLOOR = 0.0


@dataclass(frozen=True)
class AbstentionBand:
    """The window of policy scores the edition withholds regardless of the threshold."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError("an abstention band must lie inside [0, 1]")

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, scores: FloatArray) -> BoolArray:
        """Rows whose score falls inside the band."""
        return np.asarray((scores > self.low) & (scores < self.high), dtype=np.bool_)


@dataclass(frozen=True)
class OperatingPoint:
    """The frozen operating point of the policy edition."""

    threshold: float
    emission_rate: float
    takeover_point: float
    completion_point: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "emission_rate": self.emission_rate,
            "takeover_point": self.takeover_point,
            "completion_point": self.completion_point,
        }


def select_operating_point(
    curve: BenefitCurve,
    burden_ceiling: float = DEFAULT_BURDEN_CEILING,
    utility_floor: float = DEFAULT_UTILITY_FLOOR,
) -> OperatingPoint:
    """Maximise the primary contrast subject to the burden ceiling.

    Ref: Algorithm 5, step 6. Ties resolve toward the lower emission rate, because the
    criterion is stated as precision and burden together and a tie means the two settings
    are indistinguishable on utility while differing on burden.
    """
    if not curve.points:
        raise ValueError("the benefit curve is empty")
    admissible = [
        point
        for point in curve.points
        if point.emission_rate <= burden_ceiling and point.takeover.point > utility_floor
    ]
    if not admissible:
        raise ValueError("no setting of the sweep satisfies the pre-specified criterion")
    best = max(admissible, key=lambda point: (point.takeover.point, -point.emission_rate))
    return OperatingPoint(
        threshold=best.threshold,
        emission_rate=best.emission_rate,
        takeover_point=best.takeover.point,
        completion_point=best.completion.point,
    )


def band_scores(scores: FloatArray, band: AbstentionBand) -> FloatArray:
    """Score vector of the band policy: abstained windows are withheld.

    Withholding a window sets its score below any admissible threshold, which is how the
    band is expressed as a constraint on the same stream rather than as a second model.
    """
    withheld = band.contains(scores)
    adjusted = np.where(withheld, 0.0, scores)
    return np.asarray(adjusted, dtype=np.float64)


def abstention_band_comparison(
    table: PersonWindowTable,
    scores: FloatArray,
    labels: BoolArray,
    band: AbstentionBand,
    threshold: float,
) -> dict[str, object]:
    """The parity and burden comparison of Table 4 Panel D.

    Ref: Table 4 Panel D, p. 12 - "False-positive over-ride, unrestricted 3.4%" against
    "band policy 2.1%". The band policy is compared to the unrestricted policy at the same
    threshold: the comparison reports the AUROC difference with its bootstrap interval and
    the relative reduction in over-ride burden. Parity is asserted only when the interval
    covers zero.
    """
    unrestricted = np.asarray(scores, dtype=np.float64)
    banded = band_scores(scores, band)
    unrestricted_auroc = auroc(unrestricted, labels)
    banded_auroc = auroc(banded, labels)
    low, high = metric_interval(unrestricted - banded, labels, "auroc", seed=11)

    unrestricted_stream = np.asarray(table.alert, dtype=np.bool_)
    banded_stream = np.asarray(unrestricted >= threshold, dtype=np.bool_) & ~band.contains(
        unrestricted
    )
    unrestricted_burden = float(np.mean(unrestricted_stream & ~table.takeover))
    banded_burden = float(np.mean(banded_stream & ~table.takeover))
    reduction = (
        float(1.0 - banded_burden / unrestricted_burden)
        if unrestricted_burden > 0.0
        else float("nan")
    )
    return {
        "unrestricted_auroc": float(unrestricted_auroc),
        "banded_auroc": float(banded_auroc),
        "auroc_difference": float(banded_auroc - unrestricted_auroc),
        "difference_low": float(low),
        "difference_high": float(high),
        "parity": bool(low <= 0.0 <= high),
        "unrestricted_over_ride": unrestricted_burden,
        "banded_over_ride": banded_burden,
        "relative_reduction": reduction,
    }


__all__ = [
    "DEFAULT_BURDEN_CEILING",
    "DEFAULT_UTILITY_FLOOR",
    "AbstentionBand",
    "OperatingPoint",
    "abstention_band_comparison",
    "band_scores",
    "select_operating_point",
]
