"""Inference choices, multiplicity corrections and the stratum utilities."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate
from fluoroscopy_alert_takeover.inference.alternatives import (
    bootstrap_row,
    cluster_robust_rows,
    leave_one_site_out,
    solve_shift_for_marginal_rate,
    stochastic_intervention_contrast,
)
from fluoroscopy_alert_takeover.inference.bootstrap import (
    bootstrap_replicates,
    bootstrap_summary,
    normal_interval,
    percentile_interval,
)
from fluoroscopy_alert_takeover.inference.robust import (
    CLUSTER_LEVELS,
    cluster_labels,
    cluster_robust_estimate,
    cluster_robust_standard_error,
)
from fluoroscopy_alert_takeover.multiplicity.corrections import (
    benjamini_hochberg,
    holm_step_down,
    two_family_report,
)
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable
from fluoroscopy_alert_takeover.strata.gap import ceiling_report, maximum_subgroup_gap
from fluoroscopy_alert_takeover.strata.heterogeneity import (
    cochran_q,
    heterogeneity_report,
    i_squared,
    pooled_estimate,
    tau_squared,
)
from fluoroscopy_alert_takeover.strata.partitions import (
    EXPERIENCE_BANDS,
    VOLUME_CUTS,
    complexity_partition,
    experience_partition,
    pooled_complexity_partition,
    region_partition,
    silent_override_rate,
    site_partition,
    stratum_point_estimates,
    vessel_bed_partition,
    volume_tertile_partition,
)


def test_cluster_robust_standard_error_closed_form() -> None:
    influence = np.asarray([1.0, -1.0, 2.0, -2.0])
    clusters = np.asarray([0, 0, 1, 1])
    hand = np.sqrt((2.0 / 1.0) * (0.0**2 + 0.0**2) / 16.0)
    assert cluster_robust_standard_error(influence, clusters) == pytest.approx(hand, abs=1e-12)
    with pytest.raises(ValueError):
        cluster_robust_standard_error(influence, clusters[:-1])
    assert np.isnan(cluster_robust_standard_error(influence, np.zeros(4, dtype=np.int64)))


def test_cluster_labels_read_the_table(windows: PersonWindowTable) -> None:
    labels = cluster_labels(windows, "site")
    assert labels.shape[0] == windows.size
    assert np.unique(labels).shape[0] == windows.n_sites
    assert np.array_equal(cluster_labels(windows, "operator"), windows.operator_id)
    with pytest.raises(KeyError):
        cluster_labels(windows, "absent")
    assert set(CLUSTER_LEVELS) == {"site", "operator"}


def test_cluster_robust_estimate_keeps_the_point(windows: PersonWindowTable) -> None:
    rng = np.random.default_rng(21)
    estimate = InfluenceEstimate("x", 0.03, rng.normal(scale=0.05, size=windows.size))
    clustered = cluster_robust_estimate(estimate, cluster_labels(windows, "site"), "site")
    assert clustered.point == estimate.point
    assert clustered.name.endswith("cluster_site")
    assert np.isfinite(clustered.standard_error)


def test_bootstrap_replicates_and_intervals() -> None:
    draws = bootstrap_replicates(lambda rows: float(np.mean(rows)), 50, 40, 3)
    assert draws.shape == (40,)
    low, high = percentile_interval(draws)
    assert low <= high
    normal_low, normal_high = normal_interval(draws, 24.5)
    assert normal_low <= normal_high
    summary = bootstrap_summary(lambda rows: float(np.mean(rows)), 50, 24.5, 40, 3)
    assert summary["replicates"] == 40.0
    with pytest.raises(ValueError):
        bootstrap_replicates(lambda _rows: 0.0, 1, 40, 1)
    with pytest.raises(ValueError):
        bootstrap_replicates(lambda _rows: 0.0, 50, 3, 1)


def test_bootstrap_row_reproduces_the_point(windows: PersonWindowTable) -> None:
    rng = np.random.default_rng(22)
    estimate = InfluenceEstimate("x", 0.03, rng.normal(scale=0.05, size=windows.size))
    row = bootstrap_row(estimate, 60, 5)
    assert row["bootstrap_se"] > 0.0
    assert row["percentile_low"] <= row["percentile_high"]


def test_stochastic_intervention_shift_matches_the_frequency() -> None:
    rng = np.random.default_rng(23)
    propensity = np.clip(rng.uniform(0.2, 0.8, size=400), 0.01, 0.99)
    alert = (rng.uniform(size=400) < propensity).astype(np.float64)
    shift = solve_shift_for_marginal_rate(propensity, alert)
    assert float(np.mean(np.clip(propensity + shift, 0.0, 1.0))) == pytest.approx(
        float(alert.mean()), abs=0.01
    )


def test_stochastic_intervention_contrast_is_finite(windows: PersonWindowTable) -> None:
    size = windows.size
    alert = windows.alert.astype(np.float64)
    outcome = windows.outcome("takeover")
    propensity = np.clip(windows.alert_score, 0.05, 0.95)
    mu_treated = np.full(size, float(outcome[alert > 0.5].mean()))
    mu_control = np.full(size, float(outcome[alert < 0.5].mean()))
    mask = np.ones(size, dtype=np.bool_)
    estimate = stochastic_intervention_contrast(
        "shift", alert, outcome, propensity, mu_treated, mu_control, mask, 0.1
    )
    assert np.isfinite(estimate.point)
    assert estimate.influence.shape[0] == size
    assert abs(estimate.point) < 1.0


def test_cluster_robust_rows_reports_every_level(windows: PersonWindowTable) -> None:
    rng = np.random.default_rng(25)
    mask = np.ones(windows.size, dtype=np.bool_)
    estimate = InfluenceEstimate("x", 0.03, rng.normal(scale=0.05, size=windows.size))
    rows = cluster_robust_rows(estimate, windows, mask)
    assert set(rows) == {"site", "operator"}


def test_leave_one_site_out_refits_per_configuration(windows: PersonWindowTable) -> None:
    def pipeline(subset: PersonWindowTable) -> tuple[InfluenceEstimate, InfluenceEstimate]:
        size = subset.size
        estimate = InfluenceEstimate("x", float(np.mean(subset.alert)) * 0.1, np.full(size, 0.01))
        return estimate, estimate

    report = leave_one_site_out(windows, pipeline, 0)
    assert len(report["held_out"]) == windows.n_sites
    assert report["max"] >= report["min"]
    assert isinstance(report["crossed_null"], bool)
    with pytest.raises(ValueError):
        leave_one_site_out(windows, pipeline, 2)


def test_holm_step_down_hand_computed() -> None:
    values = np.asarray([0.001, 0.02, 0.2, 0.04])
    produced = holm_step_down(values, 0.05)
    assert produced.adjusted.tolist() == pytest.approx([0.004, 0.06, 0.2, 0.08])
    assert produced.rejected.tolist() == [True, False, False, False]
    assert produced.family == "holm_step_down"


def test_benjamini_hochberg_hand_computed() -> None:
    values = np.asarray([0.001, 0.02, 0.2, 0.04])
    produced = benjamini_hochberg(values, 0.05)
    assert produced.adjusted.tolist() == pytest.approx([0.004, 0.04, 0.2, 0.0533333333])
    assert produced.family == "benjamini_hochberg"
    with pytest.raises(ValueError):
        holm_step_down(np.empty(0))
    with pytest.raises(ValueError):
        benjamini_hochberg(np.empty(0))


def test_two_family_report_returns_both() -> None:
    report = two_family_report(np.asarray([0.01, 0.2]), np.asarray([0.01, 0.2, 0.3]))
    assert set(report) == {"primary", "secondary"}


def test_heterogeneity_closed_forms() -> None:
    estimates = np.asarray([1.0, 2.0, 3.0])
    errors = np.ones(3)
    q, df, p_value, pooled = cochran_q(estimates, errors)
    assert q == pytest.approx(2.0)
    assert pooled == pytest.approx(2.0)
    assert df == 2
    assert 0.0 <= p_value <= 1.0
    assert i_squared(2.0, 2) == 0.0
    assert i_squared(4.0, 2) == pytest.approx(50.0)
    assert tau_squared(2.0, 2, estimates, errors) == pytest.approx(0.0)
    report = heterogeneity_report(np.asarray([3.6, 3.1, 2.6]), np.asarray([0.77, 0.82, 0.92]))
    assert report.as_dict()["df"] == 2.0
    with pytest.raises(ValueError):
        cochran_q(np.asarray([1.0]), np.asarray([1.0]))
    with pytest.raises(ValueError):
        cochran_q(estimates, np.asarray([0.0, 0.0, 0.0]))


def test_pooled_estimate_is_size_weighted() -> None:
    produced = pooled_estimate(np.asarray([2.4, 4.6]), np.asarray([2918.0, 3441.0]))
    hand = (2.4 * 2918 + 4.6 * 3441) / (2918 + 3441)
    assert produced == pytest.approx(hand)
    with pytest.raises(ValueError):
        pooled_estimate(np.asarray([1.0, 2.0]), np.asarray([1.0]))


def test_partitions_cover_the_cohort(windows: PersonWindowTable) -> None:
    classes = complexity_partition(windows)
    assert len(classes) == 3
    assert sum(stratum.n for stratum in classes) == windows.size
    assert vessel_bed_partition(windows)[1].n == int(np.count_nonzero(windows.infrapopliteal))
    assert len(experience_partition(windows)) == len(EXPERIENCE_BANDS)
    assert len(site_partition(windows)) == windows.n_sites
    assert len(region_partition(windows)) >= 2
    pooled = pooled_complexity_partition(windows)
    assert pooled[0].n + pooled[1].n >= windows.size
    assert volume_tertile_partition(windows)[0].label == "Low tertile"
    assert VOLUME_CUTS == (420.0, 780.0)


def test_silent_override_hand_counted(windows: PersonWindowTable) -> None:
    rate = silent_override_rate(windows)
    hand = float(np.mean(windows.alert & ~windows.takeover))
    assert rate == pytest.approx(hand)
    mask = windows.complexity_class == 1
    assert silent_override_rate(windows, mask) == pytest.approx(
        float(np.mean((windows.alert & ~windows.takeover)[mask]))
    )
    with pytest.raises(ValueError):
        silent_override_rate(windows, np.zeros(windows.size, dtype=np.bool_))


def test_stratum_point_estimates_are_means(windows: PersonWindowTable) -> None:
    quantity = windows.takeover.astype(np.float64)
    produced = stratum_point_estimates(windows, complexity_partition(windows), quantity, 1.0)
    assert len(produced) == 3
    for stratum in complexity_partition(windows):
        assert produced[stratum.label] == pytest.approx(float(np.mean(quantity[stratum.mask])))


def test_subgroup_gap_and_ceiling() -> None:
    gap = maximum_subgroup_gap(["a", "b", "c"], np.asarray([2.1, 4.9, 3.0]), ceiling=2.0)
    assert gap.gap == pytest.approx(2.8)
    assert gap.lowest_label == "a" and gap.highest_label == "b"
    assert not gap.within_ceiling
    assert ceiling_report(gap)["within_ceiling"] is False
    assert maximum_subgroup_gap(["a", "b"], np.asarray([2.0, 3.0])).within_ceiling
    with pytest.raises(ValueError):
        maximum_subgroup_gap(["a"], np.asarray([1.0]))
