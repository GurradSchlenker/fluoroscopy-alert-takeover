"""Estimators: Eq. (2), the singly-robust pair, targeted learning, inference and the grid."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroscopy_alert_takeover.estimators.ablation import (
    ABLATION_GRID,
    ADJUSTMENT_SETS,
    AblationSpec,
    ablation_deficit_price,
    estimator_spread,
    nuisance_drift,
    run_ablation_grid,
)
from fluoroscopy_alert_takeover.estimators.aipw import aipw_arm_means, aipw_contrast, eq2_terms
from fluoroscopy_alert_takeover.estimators.crossfit import (
    clip_propensity,
    region_indices,
    table_design,
)
from fluoroscopy_alert_takeover.estimators.influence import (
    InfluenceEstimate,
    difference,
    linear_contrast,
    ratio,
)
from fluoroscopy_alert_takeover.estimators.nuisance import (
    LEARNERS,
    OUTCOME_LEARNERS,
    cross_fitted_outcome,
    fit_outcome,
    in_sample_outcome,
    learner_is_classifier,
)
from fluoroscopy_alert_takeover.estimators.singly_robust import (
    ipw_contrast,
    outcome_regression_contrast,
)
from fluoroscopy_alert_takeover.estimators.tmle import (
    clever_covariate,
    fit_fluctuation,
    tmle_contrast,
)
from fluoroscopy_alert_takeover.exposure.model import fit_balance_first_exposure
from fluoroscopy_alert_takeover.overlap.gate import build_certified_overlap_gate
from fluoroscopy_alert_takeover.protocol.folds import site_disjoint_patient_folds
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable


def test_eq2_reduces_to_ipw_when_the_regression_is_flat() -> None:
    rng = np.random.default_rng(7)
    alert = rng.binomial(1, 0.4, size=500).astype(np.float64)
    outcome = rng.binomial(1, 0.3, size=500).astype(np.float64)
    propensity = np.clip(rng.uniform(0.2, 0.8, size=500), 0.05, 0.95)
    flat = np.full(500, float(outcome.mean()))
    produced = eq2_terms(alert, outcome, propensity, flat, flat)
    hand = alert * (outcome - flat) / propensity - (1.0 - alert) * (outcome - flat) / (
        1.0 - propensity
    )
    assert np.allclose(produced, hand)
    with pytest.raises(ValueError):
        eq2_terms(alert, outcome[:-1], propensity, flat, flat)


def test_aipw_arm_means_agree_with_the_contrast() -> None:
    rng = np.random.default_rng(8)
    alert = rng.binomial(1, 0.5, size=300).astype(np.float64)
    outcome = rng.normal(size=300)
    propensity = np.full(300, 0.5)
    mu_treated = outcome + 1.0
    mu_control = outcome
    treated, control = aipw_arm_means(alert, outcome, propensity, mu_treated, mu_control)
    estimate = aipw_contrast("x", alert, outcome, propensity, mu_treated, mu_control)
    assert estimate.point == pytest.approx(treated - control)


def test_influence_difference_and_ratio() -> None:
    left = InfluenceEstimate("left", 2.0, np.asarray([1.0, -1.0, 0.5, -0.5]))
    right = InfluenceEstimate("right", 0.5, np.asarray([0.1, -0.1, 0.2, -0.2]))
    gap = difference("gap", left, right)
    assert gap.point == pytest.approx(1.5)
    assert np.allclose(gap.influence, left.influence - right.influence)
    fraction = ratio(left, right)
    assert fraction.point == pytest.approx(4.0)
    hand = (left.influence - 4.0 * right.influence) / 0.5
    assert np.allclose(fraction.influence, hand)
    with pytest.raises(ZeroDivisionError):
        ratio(left, InfluenceEstimate("zero", 0.0, right.influence))
    with pytest.raises(ValueError):
        difference("bad", left, InfluenceEstimate("short", 0.0, np.zeros(3)))


def test_linear_contrast_weights() -> None:
    first = InfluenceEstimate("a", 1.0, np.asarray([1.0, -1.0]))
    second = InfluenceEstimate("b", 3.0, np.asarray([0.5, -0.5]))
    combined = linear_contrast("c", [(0.75, first), (0.25, second)])
    assert combined.point == pytest.approx(1.5)
    assert np.allclose(combined.influence, 0.75 * first.influence + 0.25 * second.influence)
    with pytest.raises(ValueError):
        linear_contrast("c", [])


def test_influence_interval_and_p_value() -> None:
    rng = np.random.default_rng(9)
    estimate = InfluenceEstimate("x", 0.2, rng.normal(size=400))
    low, high = estimate.interval()
    assert low < estimate.point < high
    assert 0.0 <= estimate.p_value() <= 1.0
    assert estimate.covers(estimate.point)
    assert estimate.as_dict()["n"] == 400.0


def test_nuisance_learners_and_classifiers() -> None:
    assert set(OUTCOME_LEARNERS) == set(LEARNERS)
    assert learner_is_classifier("glm")
    assert not learner_is_classifier("random_forest")
    with pytest.raises(KeyError):
        learner_is_classifier("absent")


def test_fit_outcome_falls_back_on_a_degenerate_arm() -> None:
    design = np.random.default_rng(2).normal(size=(20, 3))
    constant = np.ones(20)
    produced = fit_outcome("glm", design[:10], constant[:10], design[10:])
    assert np.allclose(produced, 1.0)


def test_cross_fitted_outcome_is_out_of_fold(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    design = table_design(windows, None)
    alert = windows.alert.astype(np.int64)
    outcome = windows.outcome("takeover")
    mu_treated, _mu_control = cross_fitted_outcome(design, alert, outcome, folds, "glm")
    assert mu_treated.shape == outcome.shape
    assert bool(((mu_treated > 0.0) & (mu_treated < 1.0)).all())
    in_sample_treated, _in_sample_control = in_sample_outcome(design, alert, outcome, "glm")
    assert not np.allclose(mu_treated, in_sample_treated)


def test_trim_records_the_moved_mass() -> None:
    propensity = np.asarray([0.001, 0.5, 0.999])
    trimmed = clip_propensity(propensity, 0.01)
    assert trimmed.trimmed_mass == pytest.approx(2.0 / 3.0)
    assert trimmed.propensity.tolist() == [0.01, 0.5, 0.99]
    with pytest.raises(ValueError):
        clip_propensity(propensity, 0.6)


def test_clever_covariate_and_fluctuation() -> None:
    alert = np.asarray([1.0, 1.0, 0.0, 0.0])
    propensity = np.full(4, 0.5)
    covariate = clever_covariate(alert, propensity)
    assert covariate.tolist() == [2.0, 2.0, -2.0, -2.0]
    initial = np.full(4, 0.4)
    outcome = np.asarray([1.0, 0.0, 1.0, 0.0])
    epsilon = fit_fluctuation(initial, covariate, outcome)
    assert np.isfinite(epsilon)
    assert abs(epsilon) < 10.0


def test_tmle_matches_aipw_with_a_flat_regression() -> None:
    rng = np.random.default_rng(12)
    alert = rng.binomial(1, 0.5, size=600).astype(np.float64)
    outcome = rng.binomial(1, 0.35, size=600).astype(np.float64)
    propensity = np.full(600, 0.5)
    flat = np.full(600, float(outcome.mean()))
    targeted = tmle_contrast("t", alert, outcome, propensity, flat + 0.05, flat)
    plain = aipw_contrast("a", alert, outcome, propensity, flat + 0.05, flat)
    assert targeted.point == pytest.approx(plain.point, abs=0.02)


def test_singly_robust_rows_agree_when_the_model_is_right() -> None:
    rng = np.random.default_rng(13)
    alert = rng.binomial(1, 0.5, size=800).astype(np.float64)
    outcome = np.where(alert > 0.5, 0.3, 0.2) + rng.normal(scale=0.01, size=800)
    propensity = np.full(800, 0.5)
    ipw = ipw_contrast("i", alert, outcome, propensity)
    regression = outcome_regression_contrast(
        "r", np.where(alert > 0.5, 0.3, 0.2) * 0 + 0.3, np.zeros(800) + 0.2
    )
    assert ipw.point == pytest.approx(0.1, abs=0.01)
    assert regression.point == pytest.approx(0.1, abs=1e-9)


def test_ablation_specs_are_declared() -> None:
    assert len(ABLATION_GRID) == 12
    assert set(ADJUSTMENT_SETS) == {
        "full_history",
        "imaging_only",
        "workflow_only",
        "operator_only",
    }
    with pytest.raises(ValueError):
        AblationSpec("bad", "absent", "full_history")
    with pytest.raises(ValueError):
        AblationSpec("bad", "aipw", "absent")
    with pytest.raises(ValueError):
        AblationSpec("bad", "ipw", "full_history", nuisance="random_forest")


def test_ablation_grid_runs_and_measures_spread(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    trimmed = clip_propensity(exposure.propensity, 0.02)
    gate = build_certified_overlap_gate(windows, trimmed.propensity, tolerance=0.5)
    rows = run_ablation_grid(
        windows, folds, gate.mask, ("takeover", "completion"), 0.02, grid=ABLATION_GRID[:4]
    )
    assert len(rows) == 4
    assert all(np.isfinite(row.takeover.point) for row in rows)
    assert all("takeover" in row.as_dict() for row in rows)


def test_ablation_summaries_from_a_full_grid(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    trimmed = clip_propensity(exposure.propensity, 0.02)
    gate = build_certified_overlap_gate(windows, trimmed.propensity, tolerance=0.5)
    rows = run_ablation_grid(windows, folds, gate.mask, ("takeover", "completion"), 0.02)
    spread = estimator_spread(rows, "takeover")
    hand = abs(
        100.0
        * next(row for row in rows if row.spec.label == "ipw_only_full_history").takeover.point
        - 100.0
        * next(
            row for row in rows if row.spec.label == "outcome_regression_only_full_history"
        ).takeover.point
    )
    assert spread == pytest.approx(hand)
    assert nuisance_drift(rows, "takeover") >= 0.0
    assert np.isfinite(ablation_deficit_price(rows, "completion"))


def test_region_helpers(windows: PersonWindowTable) -> None:
    mask = windows.alert.copy()
    indices = region_indices(mask)
    assert indices.shape[0] == int(np.count_nonzero(mask))
    assert np.array_equal(indices, np.nonzero(mask)[0])
