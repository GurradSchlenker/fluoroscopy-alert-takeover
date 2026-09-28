"""Analytic schema, window layout and the identification probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluoroscopy_alert_takeover.cohort.builder import build_person_windows, cohort_digest
from fluoroscopy_alert_takeover.cohort.layout import CohortLayout
from fluoroscopy_alert_takeover.protocol.probes import (
    consistency_probe,
    exchangeability_probe,
    identifiable_exposure_state,
    positivity_probe,
)
from fluoroscopy_alert_takeover.protocol.schema import CASE_MIX, COVARIATE_BLOCKS, PersonWindowTable
from fluoroscopy_alert_takeover.protocol.strategies import (
    AlertPolicyEdition,
    Strategy,
    TargetTrialProtocol,
)
from fluoroscopy_alert_takeover.protocol.windowing import WindowSpec, follow_up_end, lay_windows


def test_every_block_member_is_a_column(windows: PersonWindowTable) -> None:
    for members in COVARIATE_BLOCKS.values():
        for name in members:
            assert name in windows.covariates
    for name in CASE_MIX:
        assert name in windows.covariates


def test_columns_are_aligned(windows: PersonWindowTable) -> None:
    for values in windows.covariates.values():
        assert values.shape[0] == windows.size
    assert windows.alert.shape[0] == windows.takeover.shape[0] == windows.completion.shape[0]


def test_schema_rejects_a_short_column(windows: PersonWindowTable) -> None:
    broken = dict(windows.covariates)
    broken["age_years"] = windows.covariates["age_years"][:-1]
    with pytest.raises(ValueError):
        PersonWindowTable(**{**windows.__dict__, "covariates": broken})


def test_block_and_full_design_widths(windows: PersonWindowTable) -> None:
    assert windows.design().shape[1] == sum(len(members) for members in COVARIATE_BLOCKS.values())
    assert windows.full_design().shape[1] == windows.design().shape[1] + len(CASE_MIX)
    assert windows.full_design(("imaging",)).shape[1] == len(COVARIATE_BLOCKS["imaging"]) + len(
        CASE_MIX
    )
    assert len(windows.column_names()) == windows.design().shape[1]


def test_windowing_layout() -> None:
    spec = WindowSpec(length_s=30.0, stride_s=30.0, response_s=30.0)
    windows = lay_windows(240.0, spec)
    assert windows[0] == (0.0, 30.0)
    assert windows[-1] == (210.0, 240.0)
    assert len(windows) == 8
    assert lay_windows(20.0, spec) == [(0.0, 20.0)]


def test_windowing_rejects_a_zero_stride() -> None:
    with pytest.raises(ValueError):
        lay_windows(100.0, WindowSpec(length_s=30.0, stride_s=0.0))


def test_follow_up_ends_at_the_takeover() -> None:
    assert follow_up_end(10.0, 30.0, 25.0) == 25.0
    assert follow_up_end(10.0, 30.0, float("nan")) == 40.0
    assert follow_up_end(10.0, 30.0, 900.0) == 40.0


def test_policy_edition_validates_its_band() -> None:
    with pytest.raises(ValueError):
        AlertPolicyEdition("frozen", 0.4, abstention_low=0.5, abstention_high=0.2)
    assert not AlertPolicyEdition("frozen", 0.4).abstains
    assert AlertPolicyEdition("frozen", 0.4, 0.3, 0.4).abstains


def test_protocol_arms_are_named() -> None:
    protocol = TargetTrialProtocol()
    assert protocol.arm_label(True) is Strategy.ALERT_EXPOSED
    assert protocol.arm_label(False) is Strategy.ALERT_UNEXPOSED
    assert protocol.window_multipliers == (1.0, 2.0, 3.0, 4.0)


def test_consistency_probe_on_the_table(windows: PersonWindowTable) -> None:
    probe = consistency_probe(windows, 0.5)
    assert probe.assumption == "consistency"
    assert probe.value == pytest.approx(float(np.mean(identifiable_exposure_state(windows))))
    assert probe.passed


def test_positivity_probe_reflects_the_gate() -> None:
    propensity = np.linspace(0.01, 0.99, 200)
    # At a gate of 0.4 every retained row sits at the two extremes of the overlap weight
    # 2 * pi * (1 - pi), which peaks at 0.5, so a floor above 0.48 can never be met.
    assert positivity_probe(propensity, 0.4, 0.6).verdict == "FAIL"
    assert positivity_probe(propensity, 0.4, 0.4).verdict == "PASS"
    # A wide gate keeps rows near the middle of the propensity scale, where the overlap
    # weight is close to its maximum.
    assert positivity_probe(propensity, 0.05, 0.3).verdict == "PASS"


def test_exchangeability_probe_needs_every_control_to_hold() -> None:
    covering = np.asarray([0.6, 0.7, 0.9, 0.5, 0.8])
    assert exchangeability_probe(covering, 0.05).passed
    one_failure = np.asarray([0.6, 0.7, 0.01, 0.5, 0.8])
    assert not exchangeability_probe(one_failure, 0.05).passed


def test_cohort_digest_matches_the_table(windows: PersonWindowTable) -> None:
    digest = cohort_digest(windows)
    assert digest["rows"] == float(windows.size)
    assert digest["alert_rate"] == pytest.approx(float(np.mean(windows.alert)))


def test_reader_reports_a_missing_layer(tmp_path: Path) -> None:
    layout = CohortLayout(root=tmp_path)
    assert set(layout.missing()) == {"site_registry.csv", "procedures.csv", "navigation/", "cine/"}
    with pytest.raises(FileNotFoundError):
        build_person_windows(layout, WindowSpec())
