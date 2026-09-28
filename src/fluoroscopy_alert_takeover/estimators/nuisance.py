"""Nuisance regressions: the exposure model of Eq. (2) and the outcome regression.

Ref: Eq. (2), p. 17 - ``pi(H)`` from "the balance-first exposure approach" and
``mu_a(H)`` from "the result regression over the covariate past"; Table 2, p. 8 - the
nuisance family is replaced in turn by a machine-learning alternative while the
adjustment set and the estimator are held fixed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ..protocol.folds import fold_masks
from ..utils.scaling import standardise_pair
from ..utils.types import FloatArray, IntArray

OUTCOME_LEARNERS: tuple[str, ...] = ("glm", "gradient_boosted", "random_forest")

_EPS = 1e-6


def _glm() -> Any:
    return LogisticRegression(C=1e5, solver="lbfgs", max_iter=2000)


def _gradient_boosted() -> Any:
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
    )


def _random_forest() -> Any:
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=1)


LEARNERS: dict[str, Callable[[], Any]] = {
    "glm": _glm,
    "gradient_boosted": _gradient_boosted,
    "random_forest": _random_forest,
}


def learner_is_classifier(name: str) -> bool:
    """Whether a named learner returns probabilities directly."""
    if name not in LEARNERS:
        raise KeyError(f"unknown outcome learner: {name}")
    return name == "glm"


def fit_outcome(
    learner: str, design: FloatArray, outcome: FloatArray, design_eval: FloatArray
) -> FloatArray:
    """Fit ``E[Y | H]`` on one arm and score ``design_eval``.

    The design is standardised on the fitting rows only, since the arm regressions are
    refitted per fold and the held-out rows must not inform the scaling.
    """
    if learner not in LEARNERS:
        raise KeyError(f"unknown outcome learner: {learner}")
    if np.unique(outcome).shape[0] < 2:
        # A degenerate arm has no regression to fit; the arm mean is the honest answer.
        return np.full(design_eval.shape[0], float(np.mean(outcome)), dtype=np.float64)
    scaled_design, scaled_eval = standardise_pair(design, design_eval)
    estimator = LEARNERS[learner]()
    estimator.fit(scaled_design, outcome)
    if learner_is_classifier(learner):
        probabilities = estimator.predict_proba(scaled_eval)
        scored = probabilities[:, 1]
    else:
        scored = estimator.predict(scaled_eval)
    return np.asarray(np.clip(scored, _EPS, 1.0 - _EPS), dtype=np.float64)


@dataclass(frozen=True)
class CrossFittedNuisance:
    """Out-of-fold exposure propensity and the two arm regressions."""

    propensity: FloatArray
    mu_treated: FloatArray
    mu_control: FloatArray
    folds: IntArray
    learner: str

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome_learner": self.learner,
            "n_folds": int(np.unique(self.folds).shape[0]),
            "propensity_min": float(np.min(self.propensity)),
            "propensity_max": float(np.max(self.propensity)),
        }


def cross_fitted_outcome(
    design: FloatArray,
    alert: IntArray,
    outcome: FloatArray,
    folds: IntArray,
    learner: str,
) -> tuple[FloatArray, FloatArray]:
    """Cross-fitted ``mu_1`` and ``mu_0`` on every row.

    Each arm's regression is fitted only on the rows of the other folds, so the row being
    scored never enters its own model (Algorithm 3, steps 2-5, p. 18).
    """
    n_folds = int(np.unique(folds).shape[0])
    mu_treated = np.full(design.shape[0], np.nan, dtype=np.float64)
    mu_control = np.full(design.shape[0], np.nan, dtype=np.float64)
    for held_out, fitted in fold_masks(folds, n_folds):
        treated_rows = fitted[alert[fitted] == 1]
        control_rows = fitted[alert[fitted] == 0]
        if treated_rows.size == 0 or control_rows.size == 0:
            raise ValueError("a fold left one arm empty; reduce the fold count")
        mu_treated[held_out] = fit_outcome(
            learner, design[treated_rows], outcome[treated_rows], design[held_out]
        )
        mu_control[held_out] = fit_outcome(
            learner, design[control_rows], outcome[control_rows], design[held_out]
        )
    if not (np.isfinite(mu_treated).all() and np.isfinite(mu_control).all()):
        raise RuntimeError("cross-fitting left rows unscored")
    return mu_treated, mu_control


def in_sample_outcome(
    design: FloatArray, alert: IntArray, outcome: FloatArray, learner: str
) -> tuple[FloatArray, FloatArray]:
    """``mu_1`` and ``mu_0`` fitted and scored on the same rows.

    This is the row ``AIPW, single fold (no cross-fitting)`` of Table 2, p. 8, kept as the
    ablation that shows what cross-fitting buys.
    """
    treated_rows = np.nonzero(alert == 1)[0]
    control_rows = np.nonzero(alert == 0)[0]
    if treated_rows.size == 0 or control_rows.size == 0:
        raise ValueError("both arms are needed to fit the outcome regression")
    mu_treated = fit_outcome(learner, design[treated_rows], outcome[treated_rows], design)
    mu_control = fit_outcome(learner, design[control_rows], outcome[control_rows], design)
    return mu_treated, mu_control


__all__ = [
    "LEARNERS",
    "OUTCOME_LEARNERS",
    "CrossFittedNuisance",
    "cross_fitted_outcome",
    "fit_outcome",
    "in_sample_outcome",
    "learner_is_classifier",
]
