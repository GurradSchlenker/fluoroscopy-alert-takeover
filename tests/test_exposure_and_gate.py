"""Balance, family selection, the exposure model and the certified-overlap gate."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroscopy_alert_takeover.exposure.balance import (
    balance_rows,
    block_distance,
    standardised_mean_difference,
    worst_row,
)
from fluoroscopy_alert_takeover.exposure.family import (
    FAMILIES,
    FAMILY_NAMES,
    candidate_strengths,
    family_distance,
    fit_predict,
    select_family,
)
from fluoroscopy_alert_takeover.exposure.model import (
    exposure_side_balance,
    fit_balance_first_exposure,
    in_sample_propensity,
)
from fluoroscopy_alert_takeover.overlap.effective_sample_size import (
    effective_sample_size,
    kish_effective_sample_size,
)
from fluoroscopy_alert_takeover.overlap.gate import (
    DEFAULT_GRID,
    boundary_mass,
    build_certified_overlap_gate,
    covariate_marginal_distance,
    gate_mask,
    positivity_deficit_price,
    sweep_gate,
)
from fluoroscopy_alert_takeover.protocol.folds import patient_folds, site_disjoint_patient_folds
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable


def test_smd_closed_form() -> None:
    values = np.asarray([0.0, 1.0, 10.0, 11.0])
    treated = np.asarray([False, False, True, True])
    produced = standardised_mean_difference(values, treated)
    hand = 10.0 / np.sqrt(0.5 * (0.25 + 0.25))
    assert produced == pytest.approx(hand)


def test_smd_is_zero_for_identical_arms() -> None:
    values = np.asarray([1.0, 2.0, 1.0, 2.0])
    treated = np.asarray([True, True, False, False])
    assert standardised_mean_difference(values, treated) == pytest.approx(0.0)


def test_smd_handles_a_flat_column() -> None:
    values = np.ones(4)
    treated = np.asarray([True, True, False, False])
    assert standardised_mean_difference(values, treated) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        standardised_mean_difference(np.ones(3), treated)


def test_weighted_smd_uses_the_group_weight_share() -> None:
    values = np.asarray([0.0, 1.0, 0.0, 1.0])
    treated = np.asarray([True, True, False, False])
    weights = np.asarray([1.0, 1.0, 100.0, 1.0])
    produced = standardised_mean_difference(values, treated, weights)
    exposed_mean = 0.5
    control_mean = 1.0 / 101.0
    control_var = (100.0 * (0.0 - control_mean) ** 2 + (1.0 - control_mean) ** 2) / 101.0
    exposed_var = 0.25
    hand = abs(exposed_mean - control_mean) / np.sqrt(0.5 * (exposed_var + control_var))
    assert produced == pytest.approx(hand)


def test_balance_rows_cover_every_block(windows: PersonWindowTable) -> None:
    rows = balance_rows(windows)
    assert {row.group for row in rows} == {
        "imaging",
        "kinematic",
        "workflow",
        "operator",
        "case_mix",
    }
    assert all(row.smd_before >= 0.0 for row in rows)
    assert worst_row(rows, "smd_before") in rows
    assert block_distance(rows, "smd_before", ("imaging",)) >= 0.0
    with pytest.raises(ValueError):
        block_distance(rows, "smd_before", ("absent",))


def test_every_family_fits_and_scores() -> None:
    rng = np.random.default_rng(3)
    design = rng.normal(size=(240, 5))
    treated = (rng.uniform(size=240) < 1.0 / (1.0 + np.exp(-design[:, 0]))).astype(np.int64)
    for family in FAMILY_NAMES:
        for strength in candidate_strengths(family):
            produced = fit_predict(family, strength, design, treated, design[:5])
            assert produced.shape == (5,)
            assert bool(((produced >= 0.0) & (produced <= 1.0)).all())
    with pytest.raises(KeyError):
        fit_predict("absent", 1.0, design, treated, design)
    with pytest.raises(ValueError):
        fit_predict("linear", 1.0, design, np.ones(240, dtype=np.int64), design)


def test_family_grids_are_declared() -> None:
    assert set(FAMILY_NAMES) == set(FAMILIES)
    assert candidate_strengths("linear") == (1.0,)
    assert len(candidate_strengths("regularised_logistic")) == 5


def test_select_family_picks_the_smallest_distance() -> None:
    name, scores = select_family(
        {"regularised_logistic": 0.3, "linear": 0.1, "gradient_boosted": 0.2}
    )
    assert name == "linear"
    assert scores["gradient_boosted"] == 0.2
    with pytest.raises(KeyError):
        select_family({"absent": 1.0})


def test_patient_folds_never_split_a_patient(windows: PersonWindowTable) -> None:
    folds, sizes = site_disjoint_patient_folds(windows, 4, 11)
    assert len(sizes) == 4
    for patient in np.unique(windows.patient_id):
        assert np.unique(folds[windows.patient_id == patient]).shape[0] == 1
    with pytest.raises(ValueError):
        patient_folds(windows.patient_id, 1, 11)


def test_exposure_model_freezes_a_family_and_weights(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    model = fit_balance_first_exposure(windows, folds)
    assert model.family in FAMILY_NAMES
    assert model.propensity.shape == (windows.size,)
    assert bool(((model.propensity > 0.0) & (model.propensity < 1.0)).all())
    weights = model.inverse_probability_weights(windows.alert)
    hand = np.where(windows.alert, 1.0 / model.propensity, 1.0 / (1.0 - model.propensity))
    assert np.allclose(weights, hand)
    summary = exposure_side_balance(model, windows)
    assert set(summary) == {"imaging", "kinematic", "workflow", "operator", "case_mix"}
    assert model.as_dict()["n_folds"] == 3


def test_in_sample_propensity_differs_from_the_cross_fitted_one(windows: PersonWindowTable) -> None:
    """The no-cross-fitting row must be a different propensity, not a relabelled one."""
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    cross_fitted = fit_balance_first_exposure(windows, folds)
    in_sample = in_sample_propensity(windows, cross_fitted.family, cross_fitted.strength)
    assert in_sample.shape == cross_fitted.propensity.shape
    assert bool(((in_sample > 0.0) & (in_sample < 1.0)).all())
    assert not np.allclose(in_sample, cross_fitted.propensity)


def test_kish_effective_sample_size_closed_form() -> None:
    weights = np.asarray([1.0, 1.0, 1.0, 1.0])
    assert kish_effective_sample_size(weights) == pytest.approx(4.0)
    skewed = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert kish_effective_sample_size(skewed) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        kish_effective_sample_size(np.empty(0))
    with pytest.raises(ValueError):
        kish_effective_sample_size(np.asarray([-1.0, 1.0]))


def test_effective_sample_size_on_a_gate() -> None:
    propensity = np.linspace(0.01, 0.99, 100)
    mask = gate_mask(propensity, 0.2)
    assert effective_sample_size(propensity, mask) > 0.0
    assert effective_sample_size(propensity, np.zeros(100, dtype=np.bool_)) == 0.0


def test_gate_mask_is_the_stated_region() -> None:
    propensity = np.asarray([0.05, 0.2, 0.5, 0.8, 0.95])
    assert gate_mask(propensity, 0.2).tolist() == [False, True, True, True, False]
    with pytest.raises(ValueError):
        gate_mask(propensity, 0.7)


def test_gate_sweep_is_monotone_in_retained_mass(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    model = fit_balance_first_exposure(windows, folds)
    sweep = sweep_gate(windows, model.propensity)
    assert np.all(np.diff(sweep.retained) <= 1e-12)
    assert sweep.epsilon.shape[0] == len(DEFAULT_GRID)
    assert np.all(sweep.kappa >= 0.0)
    with pytest.raises(ValueError):
        sweep_gate(windows, model.propensity, grid=(0.2, 0.1))


def test_certified_gate_selects_the_strictest_balanced_gate(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    model = fit_balance_first_exposure(windows, folds)
    loose = build_certified_overlap_gate(windows, model.propensity, tolerance=5.0)
    strict = build_certified_overlap_gate(windows, model.propensity, tolerance=0.001)
    assert loose.epsilon >= strict.epsilon
    assert not loose.deficit_flagged or strict.deficit_flagged
    assert loose.retained >= strict.retained
    assert 0.0 <= boundary_mass(model.propensity, loose.epsilon) <= 1.0
    assert loose.as_dict()["retained"] == loose.retained


def test_marginal_distance_is_infinite_without_overlap(windows: PersonWindowTable) -> None:
    propensity = np.where(windows.alert, 0.9, 0.1)
    assert np.isinf(
        covariate_marginal_distance(windows, propensity, gate_mask(propensity, 0.45), ("imaging",))
    )
    with pytest.raises(ValueError):
        covariate_marginal_distance(
            windows, np.full(windows.size, 0.5), np.ones(windows.size, dtype=np.bool_), ("absent",)
        )


def test_family_distance_rejects_a_misaligned_vector(windows: PersonWindowTable) -> None:
    with pytest.raises(ValueError):
        family_distance(np.full(3, 0.5), windows)


def test_positivity_deficit_price_is_a_difference() -> None:
    assert positivity_deficit_price(4.1, 3.2) == pytest.approx(0.9)
