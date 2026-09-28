"""Falsification, sensitivity and the operating-point sweep."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluoroscopy_alert_takeover.estimators.influence import InfluenceEstimate
from fluoroscopy_alert_takeover.falsification.bias_factor import (
    DEFAULT_GAMMA_GRID,
    bias_factor_curve,
    contrast_under_bias,
    null_crossing,
)
from fluoroscopy_alert_takeover.falsification.evalue import (
    e_value_ci_limit,
    e_value_from_risk_ratio,
    e_value_risk_difference,
    risk_ratio_from_difference,
    unexposed_risk,
)
from fluoroscopy_alert_takeover.falsification.negative_controls import (
    EXPOSURE_SIDE_CHANNEL,
    exposure_side_channel,
    negative_control_panel,
    panel_verdict,
)
from fluoroscopy_alert_takeover.falsification.protocol import run_falsification
from fluoroscopy_alert_takeover.operating_point.abstention import (
    AbstentionBand,
    abstention_band_comparison,
    band_scores,
    select_operating_point,
)
from fluoroscopy_alert_takeover.operating_point.benefit_curve import (
    DEFAULT_THRESHOLDS,
    BenefitCurve,
    BenefitPoint,
    causal_benefit_curve,
    interrupt_burden,
    peak_threshold,
    regenerate_stream,
)
from fluoroscopy_alert_takeover.protocol.schema import NEGATIVE_CONTROLS, PersonWindowTable


def _estimate(point: float, spread: float, size: int = 400) -> InfluenceEstimate:
    """A fixed estimate, so the suite has no hash-seed dependence."""
    seed = int(abs(point) * 1e6) * 1_000_003 + int(spread * 1e6) * 1009 + size
    rng = np.random.default_rng(seed)
    influence = rng.normal(scale=spread, size=size)
    return InfluenceEstimate("x", point, influence)


def test_risk_ratio_and_evalue_closed_forms() -> None:
    ratio = risk_ratio_from_difference(0.032, 0.194)
    assert ratio == pytest.approx(0.226 / 0.194)
    hand = ratio + math.sqrt(ratio * (ratio - 1.0))
    assert e_value_risk_difference(0.032, 0.194) == pytest.approx(hand)
    assert e_value_from_risk_ratio(0.5) == pytest.approx(e_value_from_risk_ratio(2.0))
    assert e_value_from_risk_ratio(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        risk_ratio_from_difference(0.1, 1.5)


def test_evalue_at_the_confidence_limit() -> None:
    low = e_value_ci_limit(0.032, (0.021, 0.043), 0.194)
    hand = e_value_risk_difference(0.021, 0.194)
    assert low == pytest.approx(hand)
    assert e_value_ci_limit(0.032, (-0.01, 0.02), 0.194) == 1.0


def test_unexposed_risk_reads_the_control_arm() -> None:
    alert = np.asarray([1.0, 0.0, 0.0, 1.0])
    outcome = np.asarray([1.0, 1.0, 0.0, 0.0])
    assert unexposed_risk(alert, outcome, np.ones(4)) == pytest.approx(0.5)
    exposed_only = np.asarray([1.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        unexposed_risk(alert, outcome, exposed_only)


def test_bias_factor_curve_and_crossing() -> None:
    crossing = null_crossing(0.032, 0.194)
    assert crossing == pytest.approx(0.226 / 0.194)
    curve = bias_factor_curve(0.032, 0.194, tuple(np.round(np.arange(1.0, 1.4, 0.01), 4)))
    assert np.all(np.diff(curve.contrast) <= 1e-12)
    assert curve.crossing == pytest.approx(crossing, abs=0.02)
    assert contrast_under_bias(0.032, 0.194, 1.0) == pytest.approx(0.032)
    with pytest.raises(ValueError):
        contrast_under_bias(0.032, 0.194, 0.5)
    assert len(DEFAULT_GAMMA_GRID) > 50


def test_negative_control_panel_and_verdict() -> None:
    def contrast(name: str) -> InfluenceEstimate:
        return _estimate(0.09 if name == "absorbing" else 0.0, 0.004, size=4000)

    rows = negative_control_panel(contrast, ("a", "b"))
    assert [row.outcome for row in rows] == ["a", "b"]
    assert panel_verdict(rows) == "PASS"
    failures = negative_control_panel(contrast, ("absorbing",))
    assert panel_verdict(failures) == "FAIL"
    del failures
    side = exposure_side_channel(contrast)
    assert side.outcome == EXPOSURE_SIDE_CHANNEL


def test_falsification_report_carries_every_probe() -> None:
    def contrast(name: str) -> InfluenceEstimate:
        return _estimate(0.0, 0.004)

    report = run_falsification(_estimate(0.032, 0.004), 0.194, contrast)
    payload = report.as_dict()
    assert payload["verdict"] in {"PASS", "FAIL"}
    assert len(payload["outcome_side"]) == len(NEGATIVE_CONTROLS)
    assert payload["e_value"] > 1.0
    assert payload["bias_factor_curve"]["null_crossing"] > 1.0


def test_exchangeability_probe_fails_when_a_control_moves() -> None:
    def contrast(name: str) -> InfluenceEstimate:
        return _estimate(0.09 if name == NEGATIVE_CONTROLS[0] else 0.0, 0.002, size=2000)

    report = run_falsification(_estimate(0.05, 0.002, size=2000), 0.2, contrast)
    assert report.verdict == "FAIL"


def test_regenerate_stream_uses_the_score_threshold(windows: PersonWindowTable) -> None:
    stream = regenerate_stream(windows, windows.alert_score, 0.4)
    assert np.array_equal(stream.alert, windows.alert_score >= 0.4)
    assert np.array_equal(stream.takeover, windows.takeover)
    with pytest.raises(ValueError):
        regenerate_stream(windows, windows.alert_score[:-1], 0.4)
    with pytest.raises(ValueError):
        regenerate_stream(windows, windows.alert_score, 1.4)


def test_benefit_curve_and_peak(windows: PersonWindowTable) -> None:
    def pipeline(regenerated: PersonWindowTable) -> tuple[InfluenceEstimate, InfluenceEstimate]:
        share = float(np.mean(regenerated.alert))
        return _estimate(share * 0.1, 0.002), _estimate(share * 0.05, 0.002)

    curve = causal_benefit_curve(windows, windows.alert_score, pipeline, (0.3, 0.4, 0.5))
    assert [point.threshold for point in curve.points] == [0.3, 0.4, 0.5]
    assert curve.as_dict()["points"][0]["threshold"] == 0.3
    assert peak_threshold(curve, "takeover") in (0.3, 0.4, 0.5)
    with pytest.raises(ValueError):
        causal_benefit_curve(windows, windows.alert_score, pipeline, (0.5, 0.3))
    with pytest.raises(KeyError):
        peak_threshold(curve, "absent")
    assert len(DEFAULT_THRESHOLDS) == 5


def test_interrupt_burden_counts_procedures(windows: PersonWindowTable) -> None:
    burden = interrupt_burden(windows, np.ones(windows.size, dtype=np.bool_))
    hand = float(np.count_nonzero(windows.alert)) / float(np.unique(windows.procedure_id).shape[0])
    assert burden == pytest.approx(hand)
    with pytest.raises(ValueError):
        interrupt_burden(windows, np.ones(3, dtype=np.bool_))


def test_operating_point_respects_the_burden_ceiling() -> None:
    points = [
        BenefitPoint(0.15, 0.9, _estimate(0.02, 0.002), _estimate(0.01, 0.002)),
        BenefitPoint(0.35, 0.5, _estimate(0.05, 0.002), _estimate(0.03, 0.002)),
        BenefitPoint(0.55, 0.2, _estimate(0.01, 0.002), _estimate(0.01, 0.002)),
    ]
    chosen = select_operating_point(BenefitCurve(points), burden_ceiling=0.6)
    assert chosen.threshold == 0.35
    with pytest.raises(ValueError):
        select_operating_point(BenefitCurve(points), burden_ceiling=0.1)


def test_abstention_band_and_comparison(windows: PersonWindowTable) -> None:
    band = AbstentionBand(0.4, 0.6)
    assert band.width == pytest.approx(0.2)
    with pytest.raises(ValueError):
        AbstentionBand(0.6, 0.4)
    withheld = band.contains(windows.alert_score)
    assert np.array_equal(band_scores(windows.alert_score, band) == 0.0, withheld)
    report = abstention_band_comparison(windows, windows.alert_score, windows.alert, band, 0.35)
    assert set(report) >= {"unrestricted_auroc", "banded_auroc", "parity", "relative_reduction"}
    assert 0.0 <= report["unrestricted_over_ride"] <= 1.0
