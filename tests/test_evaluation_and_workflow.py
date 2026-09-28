"""Discrimination, the reader comparison, the ceiling, latency and workflow metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluoroscopy_alert_takeover.cohort.layout import CohortLayout
from fluoroscopy_alert_takeover.evaluation.discrimination import (
    auprc_auc,
    auroc,
    brier_score,
    discrimination_report,
    evaluate,
    metric_interval,
    threshold_operating_point,
)
from fluoroscopy_alert_takeover.evaluation.headroom import achievable_ceiling, headroom
from fluoroscopy_alert_takeover.evaluation.reader_study import (
    ARM_NAMES,
    ArmCounts,
    load_reader_arms,
    reader_comparison,
    relative_risk,
    sensitivity_from_estimate,
)
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable
from fluoroscopy_alert_takeover.workflow.latency import (
    StageLedger,
    budget_fraction,
    observed_action_budget,
    offline_analysis_rate,
    timed,
)
from fluoroscopy_alert_takeover.workflow.procedure_metrics import (
    METRICS,
    describe,
    integration_cost,
    unadjusted_difference,
    workflow_contrast,
)


def test_discrimination_on_a_separable_stream() -> None:
    scores = np.asarray([0.1, 0.2, 0.7, 0.9])
    labels = np.asarray([False, False, True, True])
    assert auroc(scores, labels) == pytest.approx(1.0)
    assert auprc_auc(scores, labels) == pytest.approx(1.0)
    assert brier_score(scores, labels) == pytest.approx(float(np.mean((scores - labels) ** 2)))
    statistic = evaluate(scores, labels)
    assert statistic.positives == 2
    assert statistic.as_dict()["n"] == 4.0


def test_discrimination_rejects_a_single_class() -> None:
    with pytest.raises(ValueError):
        auroc(np.asarray([0.1, 0.2]), np.asarray([True, True]))
    with pytest.raises(ValueError):
        evaluate(np.asarray([]), np.asarray([], dtype=np.bool_))


def test_metric_interval_brackets_the_point(windows: PersonWindowTable) -> None:
    report = discrimination_report(windows.alert_score, windows.alert, n_replicates=40, seed=3)
    assert set(report) >= {"auroc", "auprc", "brier", "auroc_low", "auroc_high"}
    low, high = metric_interval(
        windows.alert_score, windows.alert, "auprc", n_replicates=40, seed=3
    )
    assert low <= high


def test_threshold_operating_point_matches_hand_computation() -> None:
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    labels = np.asarray([False, False, True, True])
    point = threshold_operating_point(scores, labels, 0.5)
    assert point["recall"] == pytest.approx(1.0)
    assert point["precision"] == pytest.approx(1.0)
    assert point["emission_rate"] == pytest.approx(0.5)


def test_reader_comparison_relative_risks() -> None:
    clinician = ArmCounts("clinician_alone", 300, 900, 200, 184, 831)
    alert = ArmCounts("alert_alone", 300, 900, 300, 254, 793)
    team = ArmCounts("team", 300, 900, 290, 252, 812)
    report = reader_comparison([clinician, alert, team])
    assert set(report["arms"]) == set(ARM_NAMES)
    assert report["relative_risk"]["team_vs_alert_alone"]["ratio"] == pytest.approx(
        team.sensitivity / alert.sensitivity
    )
    assert report["relative_risk"]["team_vs_clinician_alone"]["ratio"] == pytest.approx(
        team.sensitivity / clinician.sensitivity
    )
    assert "team_vs_team" not in report["relative_risk"]


def test_relative_risk_interval_is_log_scaled() -> None:
    left = ArmCounts("a", 100, 300, 60, 60, 270)
    right = ArmCounts("b", 100, 300, 50, 50, 280)
    report = relative_risk(left, right)
    assert report["ci_low"] < report["ratio"] < report["ci_high"]
    with pytest.raises(ValueError):
        ArmCounts("bad", 100, 300, 60, 101, 270)
    with pytest.raises(ValueError):
        ArmCounts("bad", 0, 300, 60, 0, 270)


def test_reader_layer_round_trip(sample_root: Path) -> None:
    arms = load_reader_arms(sample_root / "reader_study.csv")
    assert [arm.name for arm in arms] == list(ARM_NAMES)
    assert CohortLayout(root=sample_root).reader_study.is_file()
    assert sensitivity_from_estimate(0.5, 100) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        sensitivity_from_estimate(1.5, 100)


def test_headroom_bounds() -> None:
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    labels = np.asarray([False, False, True, True])
    ceiling = achievable_ceiling(scores, labels)
    assert ceiling == pytest.approx(1.0)
    bands = headroom(0.96, 0.91)
    assert bands["headroom"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        headroom(0.5, 0.9)


def test_stage_ledger_accounting() -> None:
    ledger = StageLedger()
    ledger.record("frame_ingest", 0.01, 0.4)
    ledger.record("frame_ingest", 0.02, 0.5)
    latencies = ledger.latencies()
    assert latencies[0].stage == "frame_ingest"
    assert latencies[0].median == pytest.approx(15.0)
    assert latencies[0].throughput == pytest.approx(1000.0 / 15.0)
    assert ledger.total_median_seconds(("frame_ingest",)) == pytest.approx(0.015)


def test_timed_records_the_elapsed_time() -> None:
    ledger = StageLedger()
    value = timed("scoring", ledger, lambda: 3.5)
    assert value == 3.5
    assert "scoring" in ledger.samples


def test_budget_arithmetic() -> None:
    budget = observed_action_budget(np.asarray([1600.0, 2400.0, 3800.0]))
    assert budget["median_ms"] == pytest.approx(2400.0)
    assert budget_fraction(54.7, 2400.0) == pytest.approx(0.0228, abs=0.0001)
    with pytest.raises(ValueError):
        budget_fraction(10.0, 0.0)
    with pytest.raises(ValueError):
        observed_action_budget(np.empty(0))


def test_offline_analysis_rate_is_an_accounting_identity() -> None:
    seconds = 60.0 * 18.4
    produced = offline_analysis_rate(seconds, 10000, 8)
    assert produced == pytest.approx(seconds)
    assert offline_analysis_rate(seconds, 10000, 16) == pytest.approx(2.0 * seconds)
    with pytest.raises(ValueError):
        offline_analysis_rate(seconds, 0, 8)


def test_workflow_describe_collapses_to_procedures(windows: PersonWindowTable) -> None:
    summary = describe(windows, "fluoroscopy_time_min")
    assert summary.n == int(np.unique(windows.procedure_id).shape[0])
    assert summary.sd >= 0.0
    with pytest.raises(KeyError):
        describe(windows, "absent")
    assert set(METRICS) >= {"fluoroscopy_time_min", "contrast_volume_ml"}


def test_integration_cost_reads_one_row_per_procedure(windows: PersonWindowTable) -> None:
    report = integration_cost(windows)
    assert report["n_procedures"] == float(np.unique(windows.procedure_id).shape[0])
    assert 0.0 <= report["share_with_cost"] <= 1.0


def test_unadjusted_and_adjusted_workflow_differences(windows: PersonWindowTable) -> None:
    mask = np.ones(windows.size, dtype=np.bool_)
    quantity = windows.workflow["fluoroscopy_time_min"]
    crude = unadjusted_difference(quantity, windows.alert, mask)
    assert crude["difference"] == pytest.approx(crude["exposed_mean"] - crude["control_mean"])
    adjusted = workflow_contrast(
        "adjusted",
        quantity,
        windows.alert.astype(np.float64),
        np.clip(windows.alert_score, 0.05, 0.95),
        np.where(windows.alert, quantity, 0.0),
        np.where(windows.alert, 0.0, quantity),
        mask,
    )
    assert np.isfinite(adjusted.point)
