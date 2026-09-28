"""Reporting, the success criteria and the end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluoroscopy_alert_takeover.analysis.pipeline import (
    AnalysisConfig,
    TargetEstimator,
    contrast_pair,
    describe_workflow,
    outcome_of,
    panel_sizes,
    run_analysis,
)
from fluoroscopy_alert_takeover.cohort.builder import build_person_windows
from fluoroscopy_alert_takeover.cohort.layout import CohortLayout
from fluoroscopy_alert_takeover.estimators.crossfit import clip_propensity
from fluoroscopy_alert_takeover.exposure.model import fit_balance_first_exposure
from fluoroscopy_alert_takeover.forest.causal_forest import ForestConfig
from fluoroscopy_alert_takeover.protocol.folds import site_disjoint_patient_folds
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable
from fluoroscopy_alert_takeover.protocol.windowing import WindowSpec
from fluoroscopy_alert_takeover.reporting.success import (
    GAP_CEILING_PP,
    TECHNICAL_SUCCESS_FLOOR,
    proportion_of_flag,
    success_criteria,
    wilson_interval,
)
from fluoroscopy_alert_takeover.reporting.tables import (
    cohort_characteristics,
    main_comparison_table,
    observed_outcomes,
    pooled_stratum_rows,
)


def _smoke_config() -> AnalysisConfig:
    return AnalysisConfig(
        n_folds=3,
        clip_gamma=0.02,
        gate_tolerance=0.25,
        gate_grid=(0.02, 0.06, 0.10, 0.14, 0.18),
        n_bootstrap=40,
        thresholds=(0.3, 0.4, 0.5),
        window_multipliers=(1.0, 2.0),
        forest=ForestConfig(trees=6, min_leaf=20, max_depth=2),
    )


def test_rebuilding_the_table_is_deterministic(sample_root: Path) -> None:
    layout = CohortLayout(root=sample_root)
    spec = WindowSpec(length_s=30.0, stride_s=30.0, response_s=30.0)
    first = build_person_windows(layout, spec)
    second = build_person_windows(layout, spec)
    assert np.array_equal(first.alert, second.alert)
    assert np.array_equal(first.takeover, second.takeover)
    assert (
        first.covariates["lesion_length_mm"].tolist()
        == second.covariates["lesion_length_mm"].tolist()
    )


def test_outcome_lookup_covers_outcomes_and_covariates(windows: PersonWindowTable) -> None:
    assert outcome_of(windows, "takeover").shape[0] == windows.size
    assert outcome_of(windows, "age_years").shape[0] == windows.size
    assert outcome_of(windows, "death").shape[0] == windows.size
    with pytest.raises(KeyError):
        outcome_of(windows, "absent")


def test_target_estimator_caches_the_nuisance(windows: PersonWindowTable) -> None:
    folds, _ = site_disjoint_patient_folds(windows, 3, 5)
    exposure = fit_balance_first_exposure(windows, folds)
    trimmed = clip_propensity(exposure.propensity, 0.02)
    estimator = TargetEstimator(table=windows, trimmed=trimmed, folds=folds, config=_smoke_config())
    mask = np.ones(windows.size, dtype=np.bool_)
    first = estimator.contrast("takeover", mask)
    second = estimator.contrast("takeover", mask)
    assert first.point == second.point
    assert estimator.nuisance("takeover") is estimator.nuisance("takeover")


def test_wilson_interval_brackets_the_share() -> None:
    low, high = wilson_interval(30, 100)
    assert low < 0.3 < high
    report = proportion_of_flag(np.asarray([True] * 30 + [False] * 70, dtype=np.bool_))
    assert report["rate"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        wilson_interval(3, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_success_criteria_report_every_condition(windows: PersonWindowTable) -> None:
    from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate

    completion = InfluenceEstimate("c", 0.02, np.full(400, 0.004))
    report = success_criteria(
        proportion_of_flag(windows.safety["technical_success"].astype(bool)), completion, 3.8
    )
    assert len(report["criteria"]) == 3
    assert report["criteria"][0]["threshold"] == TECHNICAL_SUCCESS_FLOOR
    assert report["criteria"][2]["threshold"] == GAP_CEILING_PP
    assert isinstance(report["all_met"], bool)


def test_success_criteria_flag_a_null_covering_interval() -> None:
    from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate

    completion = InfluenceEstimate("c", 0.0, np.full(400, 0.01))
    report = success_criteria({"rate": 0.95, "low": 0.9, "high": 0.99, "n": 100.0}, completion, 3.0)
    assert report["criteria"][1]["met"] is False


def test_cohort_characteristics_cover_the_printed_rows(windows: PersonWindowTable) -> None:
    rows = cohort_characteristics(windows)
    names = {row.characteristic for row in rows}
    assert {"age_years", "female", "complexity_class", "infrapopliteal", "region"} <= names
    assert all(row.smd_before >= 0.0 for row in rows)
    assert all(row.as_dict()["kind"] == row.kind for row in rows)


def test_observed_outcomes_split_by_exposure(windows: PersonWindowTable) -> None:
    report = observed_outcomes(windows)
    assert set(report) == {"takeover", "completion"}
    for values in report.values():
        assert 0.0 <= values["withheld"] <= 1.0
        assert 0.0 <= values["delivered"] <= 1.0


def test_pooled_stratum_rows_are_size_weighted() -> None:
    rows = pooled_stratum_rows(
        [("pooled", [0, 1])],
        np.asarray([2.4, 3.1]),
        np.asarray([2.0, 2.4]),
        np.asarray([0.52, 0.88]),
        np.asarray([2918.0, 2455.0]),
    )
    assert rows[0]["n"] == pytest.approx(2918.0 + 2455.0)
    assert rows[0]["takeover"] == pytest.approx((2.4 * 2918 + 3.1 * 2455) / 5373)
    with pytest.raises(ValueError):
        pooled_stratum_rows(
            [("pooled", [])],
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError):
        pooled_stratum_rows(
            [("pooled", [0, 5])],
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError):
        pooled_stratum_rows(
            [("pooled", [0])],
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
        )


def test_main_comparison_table_carries_every_block(windows: PersonWindowTable) -> None:
    from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate

    estimate = InfluenceEstimate("x", 0.03, np.full(windows.size, 0.004))
    record = main_comparison_table(
        windows,
        None,
        estimate,
        estimate,
        {"total": {"point": 0.026}},
        {"auroc": 0.9},
        {"available": False},
    )
    assert set(record) >= {
        "characteristics",
        "observed",
        "takeover",
        "alert_model",
        "reader_comparison",
    }
    assert record["max_smd_before"]["value"] >= record["max_smd_after"]["value"]


def test_pipeline_runs_end_to_end(windows: PersonWindowTable) -> None:
    result = run_analysis(windows, _smoke_config())
    assert np.isfinite(result.primary.point)
    assert np.isfinite(result.secondary.point)
    residual = result.paths.total.point - (result.paths.direct.point + result.paths.mediated.point)
    assert residual == pytest.approx(0.0, abs=1e-10)
    assert abs(result.theta.point - result.paths.mediated.point / result.paths.total.point) <= 1e-9
    assert len(result.strata) >= 4
    assert set(result.headline()) >= {"takeover", "completion", "decomposition", "operating_point"}
    assert result.band_comparison["parity"] in {True, False}
    assert len(result.attenuation) == 2
    assert len(result.ablation) == 12


def test_pipeline_is_deterministic(windows: PersonWindowTable) -> None:
    first = run_analysis(windows, _smoke_config())
    second = run_analysis(windows, _smoke_config())
    assert first.primary.point == second.primary.point
    assert first.theta.point == second.theta.point
    assert first.gate.epsilon == second.gate.epsilon


def test_pipeline_records_the_blocked_reader_layer(windows: PersonWindowTable) -> None:
    result = run_analysis(windows, _smoke_config())
    assert result.reader["available"] is False


def test_pipeline_accepts_the_reader_layer(windows: PersonWindowTable) -> None:
    from fluoroscopy_alert_takeover.evaluation.reader_study import ArmCounts

    arms = [
        ArmCounts("clinician_alone", 300, 900, 200, 184, 831),
        ArmCounts("alert_alone", 300, 900, 300, 254, 793),
        ArmCounts("team", 300, 900, 290, 252, 812),
    ]
    result = run_analysis(windows, _smoke_config(), reader_arms=arms)
    assert result.reader["available"] is True
    assert "team_vs_alert_alone" in result.reader["relative_risk"]


def test_pipeline_panels_and_workflow(windows: PersonWindowTable) -> None:
    sizes = panel_sizes(windows)
    assert set(sizes) >= {"complexity", "experience", "site", "volume_tertile"}
    workflow = describe_workflow(windows, np.ones(windows.size, dtype=np.bool_))
    assert set(workflow) >= {"fluoroscopy_time_min", "contrast_volume_ml", "integration_cost"}
    pair = contrast_pair(_smoke_config())
    takeover, completion = pair(windows)
    assert np.isfinite(takeover.point) and np.isfinite(completion.point)


def test_pipeline_validation_of_the_fold_count() -> None:
    with pytest.raises(ValueError):
        AnalysisConfig(n_folds=1)
    with pytest.raises(ValueError):
        AnalysisConfig(clip_gamma=0.0)
