"""The negative-control panel of Algorithm 2.

Ref: Algorithm 2, steps 1-5, p. 16::

    for each negative-control outcome Y' in N do
        estimate the alert-to-Y' contrast on O* with the primary estimator
        record pass when the interval covers the null and fail otherwise
    estimate the exposure-side negative-control channel contrast on O* and record its verdict

Table 4 Panel A, p. 11, lists the five outcome-side controls and states that they "must
show no association with alert delivery under a causal account".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..estimators.influence import InfluenceEstimate
from ..protocol.schema import NEGATIVE_CONTROLS

ContrastFunction = Callable[[str], InfluenceEstimate]

# Algorithm 2, step 5 names an exposure-side channel without saying which variable carries
# it. The release uses the pre-window acquisition count, which is fixed before the analysis
# period opens and therefore cannot be moved by the alert.
EXPOSURE_SIDE_CHANNEL = "cine_frames_prior"


@dataclass(frozen=True)
class NegativeControlRow:
    """One probe of the falsification panel."""

    outcome: str
    estimate: InfluenceEstimate
    covers_null: bool

    @property
    def passed(self) -> bool:
        return self.covers_null

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "estimate": self.estimate.as_dict(),
            "covers_null": self.covers_null,
            "passed": self.passed,
        }


def negative_control_panel(
    contrast: ContrastFunction, names: tuple[str, ...] = NEGATIVE_CONTROLS
) -> list[NegativeControlRow]:
    """Run the primary estimator on every pre-specified negative-control outcome."""
    rows: list[NegativeControlRow] = []
    for name in names:
        estimate = contrast(name)
        rows.append(
            NegativeControlRow(outcome=name, estimate=estimate, covers_null=estimate.covers(0.0))
        )
    return rows


def exposure_side_channel(contrast: ContrastFunction) -> NegativeControlRow:
    """The exposure-side probe: a pre-window quantity the alert cannot have moved."""
    estimate = contrast(EXPOSURE_SIDE_CHANNEL)
    return NegativeControlRow(
        outcome=EXPOSURE_SIDE_CHANNEL, estimate=estimate, covers_null=estimate.covers(0.0)
    )


def panel_verdict(rows: list[NegativeControlRow]) -> str:
    """PASS when every control covers the null, FAIL otherwise."""
    return "PASS" if all(row.passed for row in rows) else "FAIL"


__all__ = [
    "EXPOSURE_SIDE_CHANNEL",
    "ContrastFunction",
    "NegativeControlRow",
    "exposure_side_channel",
    "negative_control_panel",
    "panel_verdict",
]
