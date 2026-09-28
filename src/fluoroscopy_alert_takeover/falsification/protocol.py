"""The pre-specified falsification pair of Algorithm 2, end to end.

Ref: Algorithm 2, p. 16, steps 1-10. The report it returns is the Table 4 Panel A and
Panel C record: the per-probe verdicts, the E-value with its confidence-limit companion,
and the bias-factor curve with its null crossing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..estimators.influence import InfluenceEstimate
from .bias_factor import DEFAULT_GAMMA_GRID, BiasFactorCurve, bias_factor_curve
from .evalue import e_value_ci_limit, e_value_risk_difference
from .negative_controls import (
    ContrastFunction,
    NegativeControlRow,
    exposure_side_channel,
    negative_control_panel,
    panel_verdict,
)


@dataclass(frozen=True)
class FalsificationReport:
    """Everything Algorithm 2 returns."""

    panel: list[NegativeControlRow]
    exposure_side: NegativeControlRow
    e_value: float
    e_value_ci: float
    bias_curve: BiasFactorCurve
    unexposed_risk: float

    @property
    def verdict(self) -> str:
        """PASS when both the outcome-side panel and the exposure-side probe hold."""
        if panel_verdict(self.panel) == "PASS" and self.exposure_side.passed:
            return "PASS"
        return "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "outcome_side": [row.as_dict() for row in self.panel],
            "exposure_side": self.exposure_side.as_dict(),
            "e_value": self.e_value,
            "e_value_ci_limit": self.e_value_ci,
            "bias_factor_curve": self.bias_curve.as_dict(),
            "unexposed_risk": self.unexposed_risk,
        }


def run_falsification(
    primary: InfluenceEstimate,
    unexposed_risk_value: float,
    contrast: ContrastFunction,
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
) -> FalsificationReport:
    """Run every probe of the pair on the certified region."""
    panel = negative_control_panel(contrast)
    side = exposure_side_channel(contrast)
    interval = primary.interval()
    return FalsificationReport(
        panel=panel,
        exposure_side=side,
        e_value=e_value_risk_difference(primary.point, unexposed_risk_value),
        e_value_ci=e_value_ci_limit(primary.point, interval, unexposed_risk_value),
        bias_curve=bias_factor_curve(primary.point, unexposed_risk_value, gamma_grid),
        unexposed_risk=unexposed_risk_value,
    )


__all__ = ["FalsificationReport", "run_falsification"]
