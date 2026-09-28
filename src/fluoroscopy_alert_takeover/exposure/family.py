"""Candidate exposure-model families and the balance-first selection rule.

Ref: Algorithm 1, steps 2-4, p. 16 - "for each candidate family f in {regularised
logistic, gradient-boosted, linear} do compute the maximum absolute standardised mean
difference SMD_f(B); retain the family minimising max_B SMD_f(B) and freeze it".

The manuscript also states that the exposure model is selected on balance rather than on
classification accuracy (Sec. 4.4, p. 17), so the criterion here is a covariate distance
and never a score such as the AUC.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..protocol.schema import PersonWindowTable
from ..utils.scaling import standardise_pair
from ..utils.types import FloatArray, IntArray
from .balance import balance_rows, block_distance

FamilyFactory = Callable[[float], Any]

FAMILY_NAMES: tuple[str, ...] = ("regularised_logistic", "gradient_boosted", "linear")

DEFAULT_BLOCKS: tuple[str, ...] = ("imaging", "kinematic", "workflow", "operator")


def _regularised_logistic(strength: float) -> Any:
    return LogisticRegression(C=float(strength), solver="lbfgs", max_iter=2000)


def _linear(strength: float) -> Any:
    # The unpenalised member of the grid: the family Algorithm 1 calls linear.
    _ = strength
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, penalty=None)


def _gradient_boosted(strength: float) -> Any:
    return GradientBoostingClassifier(
        n_estimators=int(strength),
        max_depth=2,
        learning_rate=0.1,
        random_state=0,
    )


FAMILIES: dict[str, tuple[FamilyFactory, tuple[float, ...]]] = {
    "regularised_logistic": (_regularised_logistic, (0.01, 0.03, 0.1, 0.3, 1.0)),
    "gradient_boosted": (_gradient_boosted, (50.0, 100.0)),
    "linear": (_linear, (1.0,)),
}


def candidate_strengths(family: str) -> tuple[float, ...]:
    """Hyperparameter grid of one family."""
    if family not in FAMILIES:
        raise KeyError(f"unknown exposure-model family: {family}")
    return FAMILIES[family][1]


def family_distance(
    propensity: FloatArray,
    table: PersonWindowTable,
    block_names: tuple[str, ...] = DEFAULT_BLOCKS,
) -> float:
    """Maximum block balance distance induced by one propensity vector.

    The distance is read after overlap weighting by ``pi * (1 - pi)``, because the gate of
    Algorithm 1 acts on overlap and a family that balances the restricted region while
    distorting the tails would otherwise look better than it is.
    """
    if propensity.shape[0] != table.size:
        raise ValueError("propensity vector must be aligned with the table")
    overlap = np.asarray(propensity * (1.0 - propensity), dtype=np.float64)
    if float(overlap.sum()) <= 0.0:
        raise ValueError("propensity vector carries no overlap weight")
    rows = balance_rows(table, weights=overlap)
    return block_distance(rows, "smd_after", block_names)


def select_family(scores: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Return the family with the smallest distance together with the whole score table."""
    if not scores:
        raise ValueError("no family scores to compare")
    unknown = sorted(set(scores) - set(FAMILY_NAMES))
    if unknown:
        raise KeyError(f"unknown exposure-model families: {unknown}")
    best = min(scores, key=lambda name: scores[name])
    return best, dict(scores)


def fit_predict(
    family: str,
    strength: float,
    design: FloatArray,
    treated: IntArray,
    design_eval: FloatArray,
) -> FloatArray:
    """Fit one family member and return positive-class probabilities on ``design_eval``.

    Both matrices are standardised on the fitting design's own moments, because the
    covariate blocks carry different units and an unstandardised penalised fit would treat
    a millimetre column and a grade column as comparable.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown exposure-model family: {family}")
    if np.unique(treated).shape[0] < 2:
        raise ValueError("the exposure model needs both alert states in the fitting rows")
    factory, _ = FAMILIES[family]
    scaled_design, scaled_eval = standardise_pair(design, design_eval)
    estimator = factory(strength)
    estimator.fit(scaled_design, treated)
    probabilities = estimator.predict_proba(scaled_eval)
    return np.asarray(probabilities[:, 1], dtype=np.float64)


__all__ = [
    "DEFAULT_BLOCKS",
    "FAMILIES",
    "FAMILY_NAMES",
    "FamilyFactory",
    "candidate_strengths",
    "family_distance",
    "fit_predict",
    "select_family",
]
