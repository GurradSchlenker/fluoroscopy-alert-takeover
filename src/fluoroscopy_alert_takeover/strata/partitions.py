"""Stratum definitions of the per-condition breakdown.

Ref: Table 3, pp. 9-10 - anatomic complexity classes I-III, the infrapopliteal vessel bed,
operator experience in three bands and the site/region rows, plus the two pooled rows;
Table A3 Panel B, p. 27 - the annual-volume tertiles. Sec. 4.9, p. 21 - subgroup reporting.

The manuscript prints the strata and their boundaries; the silent-over-ride column is
printed without a definition, so the released definition below is an engineering default
and the printed values are reported NOT_RUN by the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray

EXPERIENCE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_3_years", 0.0, 3.0),
    ("3_to_8_years", 3.0, 8.0),
    ("over_8_years", 8.0, np.inf),
)

VOLUME_CUTS: tuple[float, float] = (420.0, 780.0)


@dataclass(frozen=True)
class Stratum:
    """A named subgroup with its region mask and its size."""

    group: str
    label: str
    mask: BoolArray

    @property
    def n(self) -> int:
        return int(np.count_nonzero(self.mask))


def _mask(condition: BoolArray) -> BoolArray:
    return np.asarray(condition, dtype=np.bool_)


def complexity_partition(table: PersonWindowTable) -> list[Stratum]:
    """Anatomic complexity classes I, II and III."""
    names = {1: "Class I", 2: "Class II", 3: "Class III"}
    strata = [
        Stratum("complexity", names[value], _mask(table.complexity_class == value))
        for value in (1, 2, 3)
    ]
    missing = [stratum.label for stratum in strata if stratum.n == 0]
    if missing:
        raise ValueError(f"complexity strata absent from the cohort: {missing}")
    return strata


def vessel_bed_partition(table: PersonWindowTable) -> list[Stratum]:
    """Femoropopliteal and infrapopliteal vessel beds."""
    return [
        Stratum("vessel_bed", "Femoropopliteal", _mask(~table.infrapopliteal)),
        Stratum("vessel_bed", "Infrapopliteal", _mask(table.infrapopliteal)),
    ]


def experience_partition(table: PersonWindowTable) -> list[Stratum]:
    """The three operator-experience bands of Table 3."""
    years = table.operator_experience_years
    strata: list[Stratum] = []
    for label, low, high in EXPERIENCE_BANDS:
        strata.append(Stratum("operator_experience", label, _mask((years >= low) & (years < high))))
    missing = [stratum.label for stratum in strata if stratum.n == 0]
    if missing:
        raise ValueError(f"experience bands absent from the cohort: {missing}")
    return strata


def site_partition(table: PersonWindowTable) -> list[Stratum]:
    """One stratum per site."""
    return [
        Stratum("site", site, _mask(table.site == site))
        for site in sorted({str(value) for value in table.site})
    ]


def region_partition(table: PersonWindowTable) -> list[Stratum]:
    """One stratum per region."""
    return [
        Stratum("region", region, _mask(table.region == region))
        for region in sorted({str(value) for value in table.region})
    ]


def pooled_complexity_partition(table: PersonWindowTable) -> list[Stratum]:
    """The two pooled rows of Table 3: classes I-II, and class III with infrapopliteal.

    Ref: Table 3, pp. 9-10. The printed sizes add to the analytical cohort exactly, so the
    infrapopliteal row is read as disjoint from the three complexity classes: the classes
    are the femoropopliteal strata and the infrapopliteal bed forms its own stratum.
    """
    low = _mask(table.complexity_class <= 2)
    high = _mask((table.complexity_class == 3) | table.infrapopliteal)
    return [
        Stratum("complexity_pooled", "Classes I-II pooled", low),
        Stratum("complexity_pooled", "Class III and infrapopliteal pooled", high),
    ]


def volume_tertile_partition(table: PersonWindowTable) -> list[Stratum]:
    """Site annual-volume tertiles at the printed cut points.

    Ref: Table A3 Panel B, p. 27 - "Low tertile (< 420 / yr)", "Mid tertile (420-780 / yr)",
    "High tertile (> 780 / yr)".
    """
    volume = table.annual_procedure_volume
    low, high = VOLUME_CUTS
    return [
        Stratum("volume_tertile", "Low tertile", _mask(volume < low)),
        Stratum("volume_tertile", "Mid tertile", _mask((volume >= low) & (volume <= high))),
        Stratum("volume_tertile", "High tertile", _mask(volume > high)),
    ]


def silent_override_rate(table: PersonWindowTable, mask: BoolArray | None = None) -> float:
    """Share of windows in which the alert fired and no takeover followed.

    The manuscript prints this column (Table 3, p. 9; Table A3 Panel A, p. 27; Table 4
    Panel D, p. 12) without defining it, so this reading is a release engineering default:
    the alert was delivered and the surgeon did not take over, counted over every window
    of the stratum rather than over the alerted windows alone, because that is the quantity
    that moves with the alert-emission rate in the same table.
    """
    selected = np.ones(table.size, dtype=np.bool_) if mask is None else mask
    if not bool(selected.any()):
        raise ValueError("the stratum is empty")
    over_ridden = selected & table.alert & ~table.takeover
    return float(np.mean(over_ridden[selected]))


def stratum_point_estimates(
    table: PersonWindowTable, strata: list[Stratum], estimate: FloatArray, scale: float = 100.0
) -> dict[str, float]:
    """Mean of a row-level quantity inside each stratum, on the requested scale."""
    if estimate.shape[0] != table.size:
        raise ValueError("the quantity must be aligned with the table")
    return {
        stratum.label: float(scale * np.mean(estimate[stratum.mask]))
        for stratum in strata
        if stratum.n > 0
    }


__all__ = [
    "EXPERIENCE_BANDS",
    "VOLUME_CUTS",
    "Stratum",
    "complexity_partition",
    "experience_partition",
    "pooled_complexity_partition",
    "region_partition",
    "silent_override_rate",
    "site_partition",
    "stratum_point_estimates",
    "vessel_bed_partition",
    "volume_tertile_partition",
]
