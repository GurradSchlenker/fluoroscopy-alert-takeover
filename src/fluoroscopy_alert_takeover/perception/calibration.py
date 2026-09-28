"""Calibration of the alert policy, per site.

Ref: Sec. 4.7, p. 19 - "The slopes and intercepts of the calibration for each site are
presented alongside the discrimination", because discrimination alone does not show that
the alert is usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils.types import BoolArray, FloatArray, IntArray

_EPS = 1e-6


def logit(probability: FloatArray) -> FloatArray:
    """Log-odds with the probability kept away from the two boundaries."""
    clipped = np.clip(probability, _EPS, 1.0 - _EPS)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


@dataclass(frozen=True)
class CalibrationFit:
    """One site's recalibration line on the log-odds scale."""

    site: str
    slope: float
    intercept: float
    n: int

    @property
    def perfect(self) -> bool:
        return abs(self.slope - 1.0) < 0.1 and abs(self.intercept) < 0.1


@dataclass(frozen=True)
class CalibrationCurve:
    """Binned reliability curve."""

    predicted: FloatArray
    observed: FloatArray
    counts: IntArray

    @property
    def expected_error(self) -> float:
        total = float(self.counts.sum())
        if total == 0.0:
            return float("nan")
        return float(np.sum(self.counts * np.abs(self.observed - self.predicted)) / total)


def calibration_curve(scores: FloatArray, labels: BoolArray, n_bins: int = 10) -> CalibrationCurve:
    """Reliability curve with equal-count bins on the score scale."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be aligned")
    if n_bins < 2:
        raise ValueError("a reliability curve needs at least two bins")
    order = np.argsort(scores, kind="stable")
    chunks = np.array_split(order, n_bins)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        predicted.append(float(np.mean(scores[chunk])))
        observed.append(float(np.mean(labels[chunk].astype(np.float64))))
        counts.append(int(chunk.size))
    return CalibrationCurve(
        predicted=np.asarray(predicted, dtype=np.float64),
        observed=np.asarray(observed, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
    )


def recalibrate(scores: FloatArray, labels: BoolArray, groups: np.ndarray) -> list[CalibrationFit]:
    """Fit one logistic recalibration per site on the log-odds of the policy score."""
    if not (scores.shape[0] == labels.shape[0] == groups.shape[0]):
        raise ValueError("scores, labels and groups must be aligned")
    fits: list[CalibrationFit] = []
    for site in sorted({str(value) for value in groups}):
        mask = np.asarray([str(value) == site for value in groups], dtype=np.bool_)
        selected = labels[mask].astype(np.int64)
        if int(mask.sum()) < 10 or len(np.unique(selected)) < 2:
            fits.append(
                CalibrationFit(
                    site=site, slope=float("nan"), intercept=float("nan"), n=int(mask.sum())
                )
            )
            continue
        design = logit(scores[mask]).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(design, selected)
        fits.append(
            CalibrationFit(
                site=site,
                slope=float(model.coef_.ravel()[0]),
                intercept=float(model.intercept_.ravel()[0]),
                n=int(mask.sum()),
            )
        )
    return fits


def applied_recalibration(scores: FloatArray, fit: CalibrationFit) -> FloatArray:
    """Map raw policy scores through one site's recalibration line."""
    adjusted = fit.slope * logit(scores) + fit.intercept
    return np.asarray(1.0 / (1.0 + np.exp(-adjusted)), dtype=np.float64)


__all__ = [
    "CalibrationCurve",
    "CalibrationFit",
    "applied_recalibration",
    "calibration_curve",
    "logit",
    "recalibrate",
]
