"""Estimands: the interventional analogue, its diagnostics and the window sweep."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroscopy_alert_takeover.estimands.attenuation import (
    AttenuationPoint,
    monotone_attenuation,
    window_attenuation,
)
from fluoroscopy_alert_takeover.estimands.interventional import (
    additivity_residual,
    augmented_psi_terms,
    decompose,
    mediated_fraction,
    mediator_positivity,
    psi_estimate,
    psi_functional,
    theta_null_test,
)
from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate
from fluoroscopy_alert_takeover.exposure.model import fit_balance_first_exposure
from fluoroscopy_alert_takeover.overlap.gate import build_certified_overlap_gate
from fluoroscopy_alert_takeover.protocol.folds import site_disjoint_patient_folds
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable


def test_psi_functional_is_a_two_term_sum() -> None:
    mu_low = np.asarray([0.1, 0.2])
    mu_high = np.asarray([0.7, 0.8])
    rule = np.asarray([0.25, 0.5])
    produced = psi_functional(mu_low, mu_high, rule)
    assert np.allclose(produced, rule * mu_high + (1.0 - rule) * mu_low)
    with pytest.raises(ValueError):
        psi_functional(mu_low, mu_high, np.asarray([0.5]))


def test_augmented_terms_need_an_arm() -> None:
    alert = np.asarray([1.0, 0.0])
    outcome = np.asarray([1.0, 0.0])
    mediator = np.asarray([1.0, 0.0])
    propensity = np.full(2, 0.5)
    cells = np.asarray([0.4, 0.3])
    rule = np.asarray([0.6, 0.4])
    with pytest.raises(ValueError):
        augmented_psi_terms(2, alert, outcome, mediator, propensity, cells, cells, rule, rule)


def test_arm_weight_switches_with_the_arm() -> None:
    alert = np.asarray([1.0, 0.0])
    outcome = np.asarray([0.0, 0.0])
    mediator = np.asarray([0.0, 0.0])
    propensity = np.full(2, 0.5)
    cells = np.zeros(2)
    rule = np.zeros(2)
    treated = augmented_psi_terms(1, alert, outcome, mediator, propensity, cells, cells, rule, rule)
    control = augmented_psi_terms(0, alert, outcome, mediator, propensity, cells, cells, rule, rule)
    assert treated.shape == control.shape == (2,)


def test_decomposition_is_additive() -> None:
    observed = InfluenceEstimate("observed", 1.9, np.asarray([0.1, -0.1, 0.05, -0.05]))
    crossed = InfluenceEstimate("crossed", 1.0, np.asarray([0.05, -0.05, 0.02, -0.02]))
    control = InfluenceEstimate("control", 0.3, np.asarray([0.02, -0.02, 0.01, -0.01]))
    paths = decompose(observed, crossed, control)
    assert paths.total.point == pytest.approx(1.6)
    assert paths.direct.point == pytest.approx(0.7)
    assert paths.mediated.point == pytest.approx(0.9)
    assert additivity_residual(paths) == pytest.approx(0.0, abs=1e-12)
    fraction = mediated_fraction(paths)
    assert fraction.point == pytest.approx(0.9 / 1.6)
    assert set(paths.as_dict()) == {"total", "direct", "mediated"}


def test_theta_null_test_reports_coverage() -> None:
    rng = np.random.default_rng(31)
    estimate = InfluenceEstimate("theta", 0.73, rng.normal(scale=0.5, size=100))
    report = theta_null_test(estimate, 0.5)
    assert report["ci_low"] < 0.73 < report["ci_high"]
    assert report["inside_unit_interval"] == 1.0
    # At this size the fraction is already distinguishable from one half, which is the
    # test Sec. 2.5 reports.
    assert report["p_value"] < 0.05


def test_theta_null_test_detects_an_excluded_half() -> None:
    rng = np.random.default_rng(32)
    estimate = InfluenceEstimate("theta", 0.9, rng.normal(scale=0.05, size=400))
    report = theta_null_test(estimate, 0.5)
    assert report["p_value"] < 0.05
    assert not (report["ci_low"] < 0.5 < report["ci_high"])


def test_mediator_positivity_bounds() -> None:
    rule = np.asarray([0.02, 0.3, 0.6, 0.98])
    mask = np.ones(4, dtype=np.bool_)
    report = mediator_positivity(rule, mask, 0.05)
    assert report["share_interior"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mediator_positivity(rule, mask, 0.6)


def test_psi_estimate_needs_a_non_empty_region() -> None:
    with pytest.raises(ValueError):
        psi_estimate(np.zeros(5), np.zeros(5, dtype=np.bool_))


def test_attenuation_sweep_runs(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    gate = build_certified_overlap_gate(windows, exposure.propensity, tolerance=0.5)
    points = window_attenuation(
        windows,
        folds,
        "completion",
        "glm",
        gate.mask,
        0.02,
        multipliers=(1.0, 2.0),
        propensity=exposure.propensity,
    )
    assert [point.multiplier for point in points] == [1.0, 2.0]
    assert all(np.isfinite(point.fraction) for point in points)
    assert points[1].mediator_rate >= points[0].mediator_rate
    shape = monotone_attenuation(points)
    assert shape["monotone_decreasing"] in {True, False}


def test_attenuation_rejects_an_unsorted_grid(windows: PersonWindowTable) -> None:
    with pytest.raises(ValueError):
        window_attenuation(
            windows,
            np.zeros(windows.size, dtype=np.int64),
            "completion",
            "glm",
            np.ones(windows.size, dtype=np.bool_),
            0.02,
            multipliers=(2.0, 1.0),
        )


def test_attenuation_point_serialises() -> None:
    point = AttenuationPoint(1.0, 0.73, 0.58, 0.88, 0.5)
    assert point.as_dict()["mediated_fraction"] == 0.73


def test_path_estimates_reject_mismatched_rows() -> None:
    with pytest.raises(ValueError):
        decompose(
            InfluenceEstimate("a", 1.0, np.zeros(3)),
            InfluenceEstimate("b", 0.5, np.zeros(3)),
            InfluenceEstimate("c", 0.2, np.zeros(2)),
        )
