"""The honest-split causal forest and its gradient test."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroscopy_alert_takeover.estimands.interventional import augmented_psi_terms
from fluoroscopy_alert_takeover.estimators.decomposition import cross_fitted_mediation_nuisance
from fluoroscopy_alert_takeover.exposure.model import fit_balance_first_exposure
from fluoroscopy_alert_takeover.forest.causal_forest import (
    ForestConfig,
    HonestSplitCausalForest,
    forest_strata,
    forest_stratum_estimate,
    tree_stratum_estimate,
)
from fluoroscopy_alert_takeover.forest.gradient import monotone_gradient_test
from fluoroscopy_alert_takeover.forest.honest_split import honest_split
from fluoroscopy_alert_takeover.overlap.gate import build_certified_overlap_gate
from fluoroscopy_alert_takeover.protocol.folds import site_disjoint_patient_folds
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable
from fluoroscopy_alert_takeover.strata.partitions import complexity_partition


def test_honest_split_keeps_patients_whole(windows: PersonWindowTable) -> None:
    region = np.ones(windows.size, dtype=np.bool_)
    split = honest_split(windows.patient_id, region, 3)
    assert not bool((split.structural & split.estimation).any())
    assert split.structural_rows + split.estimation_rows == windows.size
    for patient in np.unique(windows.patient_id):
        rows = windows.patient_id == patient
        assert not bool((split.structural[rows] & split.estimation[rows]).any())
    with pytest.raises(ValueError):
        honest_split(windows.patient_id[:1], np.ones(1, dtype=np.bool_), 1)


def test_forest_config_validates() -> None:
    with pytest.raises(ValueError):
        ForestConfig(trees=0)
    with pytest.raises(ValueError):
        ForestConfig(min_leaf=1)


def test_forest_fits_and_leaves_are_reachable(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    gate = build_certified_overlap_gate(windows, exposure.propensity, tolerance=0.5)
    design = windows.full_design()
    forest = HonestSplitCausalForest(ForestConfig(trees=6, min_leaf=20, max_depth=2)).fit(
        design,
        windows.alert.astype(np.float64),
        windows.outcome("completion"),
        gate.mask,
        windows.patient_id,
    )
    assert len(forest.trees) == 6
    assert forest.split.estimation_rows > 0
    with pytest.raises(ValueError):
        forest_stratum_estimate(
            HonestSplitCausalForest(ForestConfig(trees=2)),
            design,
            np.zeros(windows.size),
            np.zeros(windows.size),
            gate.mask,
        )


def test_forest_strata_produces_a_fraction_per_class(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    gate = build_certified_overlap_gate(windows, exposure.propensity, tolerance=0.5)
    nuisance = cross_fitted_mediation_nuisance(
        windows, exposure.propensity, folds, "completion", "glm"
    )
    alert = windows.alert.astype(np.float64)
    outcome = windows.outcome("completion")
    mediator = windows.hand_back.astype(np.float64)
    total = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        exposure.propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_exposed,
        nuisance.mediator_exposed,
    ) - augmented_psi_terms(
        0,
        alert,
        outcome,
        mediator,
        exposure.propensity,
        nuisance.cell(0, 0),
        nuisance.cell(0, 1),
        nuisance.mediator_control,
        nuisance.mediator_control,
    )
    mediated = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        exposure.propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_exposed,
        nuisance.mediator_exposed,
    ) - augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        exposure.propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_control,
        nuisance.mediator_exposed,
    )
    design = windows.full_design()
    forest = HonestSplitCausalForest(ForestConfig(trees=6, min_leaf=20, max_depth=2)).fit(
        design, alert, outcome, gate.mask, windows.patient_id
    )
    strata = [(stratum.label, stratum.mask) for stratum in complexity_partition(windows)]
    fractions = forest_strata(forest, design, total, mediated, strata)
    assert fractions
    for estimate in fractions.values():
        assert np.isfinite(estimate.point)
        assert estimate.influence.shape[0] > 1


def test_tree_stratum_estimate_needs_rows() -> None:
    from fluoroscopy_alert_takeover.forest.causal_forest import TreeNode

    leaf = TreeNode()
    estimate = tree_stratum_estimate(
        leaf,
        np.zeros((3, 2)),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3, dtype=np.bool_),
        np.ones(3, dtype=np.bool_),
    )
    assert estimate is None


def test_gradient_test_recovers_a_hand_computed_slope() -> None:
    points = [0.2, 0.5, 0.8]
    errors = [0.1, 0.1, 0.1]
    produced = monotone_gradient_test(["a", "b", "c"], points, errors)
    order = np.asarray([0.0, 1.0, 2.0])
    centred = order - order.mean()
    hand = float(np.sum(centred * np.asarray(points)) / np.sum(centred**2))
    assert produced.slope == pytest.approx(hand)
    assert produced.monotone
    assert produced.as_dict()["monotone"] is True


def test_gradient_test_validates_inputs() -> None:
    with pytest.raises(ValueError):
        monotone_gradient_test(["a"], [0.2], [0.1])
    with pytest.raises(ValueError):
        monotone_gradient_test(["a", "b"], [0.2, 0.3], [-0.1, 0.1])
    with pytest.raises(ValueError):
        monotone_gradient_test(["a", "b"], [0.2, 0.3], [0.1, 0.1], spacing=[1.0, 1.0])


def test_gradient_variance_is_the_weighted_sum() -> None:
    produced = monotone_gradient_test(["a", "b", "c"], [0.0, 1.0, 2.0], [0.1, 0.2, 0.3])
    order = np.asarray([0.0, 1.0, 2.0])
    centred = (order - order.mean()) / np.sum((order - order.mean()) ** 2)
    hand = float(np.sqrt(np.sum(centred**2 * np.asarray([0.1, 0.2, 0.3]) ** 2)))
    assert produced.standard_error == pytest.approx(hand)
