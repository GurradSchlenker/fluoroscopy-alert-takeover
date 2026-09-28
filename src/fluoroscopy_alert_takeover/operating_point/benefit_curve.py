"""The causal benefit curve over the operating-threshold sweep.

Ref: Algorithm 5, p. 20, steps 1-5 and 8::

    for b = 1 to B do
        regenerate the alert stream from the frozen policy at threshold tau_b
        re-estimate the causal contrast on the certified region of that stream
        record CB(tau_b) and the interrupts per procedure at tau_b
    return CB(tau), tau_op and the burden profile

Table A3 Panel A, p. 27, is the printed curve: the emission rate falls from 88.1% to 26.2%
across the sweep while the contrast is deliberately non-monotone and peaks in the middle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..estimators.influence import InfluenceEstimate
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.55)

CausalPipeline = Callable[[PersonWindowTable], tuple[InfluenceEstimate, InfluenceEstimate]]


@dataclass(frozen=True)
class BenefitPoint:
    """One setting of the sweep."""

    threshold: float
    emission_rate: float
    takeover: InfluenceEstimate
    completion: InfluenceEstimate

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "emission_rate": self.emission_rate,
            "takeover": self.takeover.as_dict(),
            "completion": self.completion.as_dict(),
        }


@dataclass(frozen=True)
class BenefitCurve:
    """The whole sweep with its selected operating point."""

    points: list[BenefitPoint]

    @property
    def threshold_grid(self) -> list[float]:
        return [point.threshold for point in self.points]

    def as_dict(self) -> dict[str, object]:
        return {"points": [point.as_dict() for point in self.points]}


def regenerate_stream(
    table: PersonWindowTable, scores: FloatArray, threshold: float
) -> PersonWindowTable:
    """Replace the exposure state with the frozen policy's stream at one threshold.

    Ref: Algorithm 5, step 2, p. 20. Outcomes, covariates and the mediator are untouched;
    only the exposure column moves, which is what makes the sweep a statement about the
    policy rather than about the analysis.
    """
    if scores.shape[0] != table.size:
        raise ValueError("the score vector must be aligned with the table")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("a threshold must lie in [0, 1]")
    stream = np.asarray(scores >= threshold, dtype=np.bool_)
    return replace(table, alert=stream)


def causal_benefit_curve(
    table: PersonWindowTable,
    scores: FloatArray,
    pipeline: CausalPipeline,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> BenefitCurve:
    """Evaluate the pipeline on the regenerated stream at every threshold of the sweep."""
    if sorted(thresholds) != list(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("the threshold grid must be strictly increasing")
    points: list[BenefitPoint] = []
    for threshold in thresholds:
        stream = regenerate_stream(table, scores, threshold)
        takeover, completion = pipeline(stream)
        points.append(
            BenefitPoint(
                threshold=float(threshold),
                emission_rate=float(np.mean(stream.alert)),
                takeover=takeover,
                completion=completion,
            )
        )
    return BenefitCurve(points=points)


def peak_threshold(curve: BenefitCurve, outcome: str = "takeover") -> float:
    """Threshold at which the contrast is largest, in percentage points."""
    if not curve.points:
        raise ValueError("the curve is empty")
    if outcome == "takeover":
        return max(curve.points, key=lambda point: point.takeover.point).threshold
    if outcome == "completion":
        return max(curve.points, key=lambda point: point.completion.point).threshold
    raise KeyError(f"unknown outcome of the benefit curve: {outcome}")


def interrupt_burden(table: PersonWindowTable, mask: BoolArray) -> float:
    """Alerts per procedure inside a region, which is the burden term of Algorithm 5.

    A procedure that contributes several windows contributes several interrupts, so the
    burden is the number of alerts divided by the number of distinct procedures, not by the
    number of windows.
    """
    if mask.shape[0] != table.size:
        raise ValueError("the mask must be aligned with the table")
    procedures = np.unique(table.procedure_id[mask])
    if procedures.shape[0] == 0:
        return 0.0
    alerts = float(np.count_nonzero(table.alert[mask]))
    return alerts / float(procedures.shape[0])


__all__ = [
    "DEFAULT_THRESHOLDS",
    "BenefitCurve",
    "BenefitPoint",
    "CausalPipeline",
    "causal_benefit_curve",
    "interrupt_burden",
    "peak_threshold",
    "regenerate_stream",
]
