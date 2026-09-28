"""The balance-first regularised exposure model.

Ref: Algorithm 1, steps 1-4, p. 16 - "fit a balance-first regularised exposure model pi
over H with patient-level folds"; Sec. 4.4, p. 17 - the model "uses a regression of a
covariate with values determined by available information about the particular families
chosen based on a previous confirmatory analysis", and the family is chosen to balance
rather than to classify.

The propensity is always cross-fitted: a row's own outcome and exposure never enter the
model that scores it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocol.folds import fold_masks
from ..protocol.schema import PersonWindowTable
from ..utils.logging import get_logger
from ..utils.types import FloatArray, IntArray
from .balance import standardised_mean_difference
from .family import (
    DEFAULT_BLOCKS,
    FAMILIES,
    candidate_strengths,
    family_distance,
    fit_predict,
    select_family,
)

_LOG = get_logger("exposure.model")

_EPS = 1e-6


@dataclass(frozen=True)
class ExposureModel:
    """A frozen exposure-model family with its cross-fitted propensity vector."""

    family: str
    strength: float
    propensity: FloatArray
    family_scores: dict[str, float]
    folds: IntArray

    def inverse_probability_weights(self, alert: np.ndarray) -> FloatArray:
        """Inverse-probability weights of the observed exposure state."""
        clipped = np.clip(self.propensity, _EPS, 1.0 - _EPS)
        return np.asarray(np.where(alert, 1.0 / clipped, 1.0 / (1.0 - clipped)), dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "strength": self.strength,
            "family_scores": dict(self.family_scores),
            "n_folds": int(np.unique(self.folds).shape[0]),
        }


def _cross_fitted_propensity(
    design: FloatArray,
    treated: IntArray,
    folds: IntArray,
    family: str,
    strength: float,
) -> FloatArray:
    """Out-of-fold propensity for every row."""
    n_folds = int(np.unique(folds).shape[0])
    propensity = np.full(design.shape[0], np.nan, dtype=np.float64)
    for held_out, fitted in fold_masks(folds, n_folds):
        propensity[held_out] = fit_predict(
            family, strength, design[fitted], treated[fitted], design[held_out]
        )
    if not np.isfinite(propensity).all():
        raise RuntimeError("cross-fitting left rows unscored")
    return propensity


def fit_balance_first_exposure(
    table: PersonWindowTable,
    folds: IntArray,
    block_names: tuple[str, ...] = DEFAULT_BLOCKS,
) -> ExposureModel:
    """Fit every candidate family member, then freeze the one that balances best."""
    if folds.shape[0] != table.size:
        raise ValueError("fold assignment must be aligned with the table")
    design = table.full_design()
    treated = table.alert.astype(np.int64)

    per_family: dict[str, tuple[float, FloatArray]] = {}
    for family in FAMILIES:
        best_strength = candidate_strengths(family)[0]
        best_distance = float("inf")
        best_propensity = np.full(table.size, np.nan, dtype=np.float64)
        for strength in candidate_strengths(family):
            propensity = _cross_fitted_propensity(design, treated, folds, family, strength)
            distance = family_distance(propensity, table, block_names)
            if distance < best_distance:
                best_distance = distance
                best_strength = strength
                best_propensity = propensity
        per_family[family] = (best_strength, best_propensity)
        _LOG.info(
            "family %s: max block distance %.4f at strength %.3g",
            family,
            best_distance,
            best_strength,
        )

    scores = {
        name: family_distance(value[1], table, block_names) for name, value in per_family.items()
    }
    best_family, _ = select_family(scores)
    strength, propensity = per_family[best_family]
    return ExposureModel(
        family=best_family,
        strength=strength,
        propensity=propensity,
        family_scores=scores,
        folds=folds,
    )


def in_sample_propensity(table: PersonWindowTable, family: str, strength: float) -> FloatArray:
    """Propensity fitted and scored on the same rows, for the no-cross-fitting ablation."""
    design = table.full_design()
    treated = table.alert.astype(np.int64)
    return fit_predict(family, strength, design, treated, design)


def exposure_side_balance(model: ExposureModel, table: PersonWindowTable) -> dict[str, float]:
    """Balance summary of the frozen exposure model, written into the analysis record."""
    weights = model.inverse_probability_weights(table.alert)
    groups = table.balance_groups()
    summary: dict[str, float] = {}
    for group, columns in groups.items():
        summary[group] = float(
            max(
                standardised_mean_difference(table.covariates[column], table.alert, weights)
                for column in columns
            )
        )
    return summary


__all__ = [
    "ExposureModel",
    "exposure_side_balance",
    "fit_balance_first_exposure",
    "in_sample_propensity",
]
