"""Discrimination and calibration metrics of the alert model.

Ref: Table 1, p. 7 - AUROC 0.918 (95% CI 0.909 to 0.927), AUPRC 0.674 (95% CI 0.661 to
0.687), and the Brier score at the frozen threshold; Table A2, p. 26 - the per-site Dice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.types import BoolArray, FloatArray


@dataclass(frozen=True)
class Discrimination:
    """AUROC, AUPRC and the Brier score of one scored stream."""

    auroc: float
    auprc: float
    brier: float
    n: int
    positives: int

    def as_dict(self) -> dict[str, float]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "brier": self.brier,
            "n": float(self.n),
            "positives": float(self.positives),
        }


def auroc(scores: FloatArray, labels: BoolArray) -> float:
    """Area under the receiver-operating-characteristic curve."""
    values, targets = _prepare(scores, labels)
    return float(roc_auc_score(targets, values))


def auprc_auc(scores: FloatArray, labels: BoolArray) -> float:
    """Area under the precision-recall curve."""
    values, targets = _prepare(scores, labels)
    return float(average_precision_score(targets, values))


def brier_score(scores: FloatArray, labels: BoolArray) -> float:
    """Brier score of the scored stream."""
    values, targets = _prepare(scores, labels)
    return float(brier_score_loss(targets, values))


def _prepare(scores: FloatArray, labels: BoolArray) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(scores, dtype=np.float64)
    targets = labels.astype(np.float64)
    if values.shape != targets.shape:
        raise ValueError("scores and labels must be aligned")
    if values.size == 0:
        raise ValueError("discrimination needs a non-empty scored stream")
    if np.unique(targets).shape[0] < 2:
        raise ValueError("discrimination needs both classes in the scored stream")
    return values, targets


def evaluate(scores: FloatArray, labels: BoolArray) -> Discrimination:
    """Compute the three reported quantities at once."""
    values, targets = _prepare(scores, labels)
    return Discrimination(
        auroc=float(roc_auc_score(targets, values)),
        auprc=float(average_precision_score(targets, values)),
        brier=float(brier_score_loss(targets, values)),
        n=int(values.size),
        positives=int(np.count_nonzero(targets > 0.5)),
    )


def _bootstrap_metric(
    values: FloatArray, targets: FloatArray, metric: str, n_replicates: int, seed: int
) -> FloatArray:
    """Percentile bootstrap of one ranking metric over the scored stream."""
    generator = np.random.default_rng(seed)
    draws = np.empty(n_replicates, dtype=np.float64)
    size = values.size
    for replicate in range(n_replicates):
        picked = generator.integers(0, size, size=size)
        resampled_targets = targets[picked]
        if np.unique(resampled_targets).shape[0] < 2:
            draws[replicate] = float("nan")
            continue
        draws[replicate] = (
            roc_auc_score(resampled_targets, values[picked])
            if metric == "auroc"
            else average_precision_score(resampled_targets, values[picked])
        )
    return draws


def metric_interval(
    scores: FloatArray,
    labels: BoolArray,
    metric: str,
    level: float = 0.95,
    n_replicates: int = 200,
    seed: int = 7,
) -> tuple[float, float]:
    """Percentile bootstrap interval of AUROC or AUPRC on the scored stream."""
    values, targets = _prepare(scores, labels)
    draws = _bootstrap_metric(values, targets, metric, n_replicates, seed)
    finite = draws[np.isfinite(draws)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = 0.5 * (1.0 - level)
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def discrimination_report(
    scores: FloatArray,
    labels: BoolArray,
    level: float = 0.95,
    n_replicates: int = 200,
    seed: int = 7,
) -> dict[str, float]:
    """The Table 1 alert-model performance block for one scored stream."""
    statistic = evaluate(scores, labels)
    auroc_low, auroc_high = metric_interval(scores, labels, "auroc", level, n_replicates, seed)
    pr_low, pr_high = metric_interval(scores, labels, "auprc", level, n_replicates, seed)
    return {
        **statistic.as_dict(),
        "auroc_low": auroc_low,
        "auroc_high": auroc_high,
        "auprc_low": pr_low,
        "auprc_high": pr_high,
    }


def threshold_operating_point(
    scores: FloatArray, labels: BoolArray, threshold: float
) -> dict[str, float]:
    """Recall and precision at a frozen operating threshold.

    Ref: Sec. 2.9, p. 6 - "While the recall of alert was 0.847, the precision was 0.684."
    """
    values, targets = _prepare(scores, labels)
    predicted = (values >= threshold).astype(np.int64)
    truth = (targets > 0.5).astype(np.int64)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "emission_rate": float(np.mean(predicted)),
    }


__all__ = [
    "Discrimination",
    "auprc_auc",
    "auroc",
    "brier_score",
    "discrimination_report",
    "evaluate",
    "metric_interval",
    "threshold_operating_point",
]
