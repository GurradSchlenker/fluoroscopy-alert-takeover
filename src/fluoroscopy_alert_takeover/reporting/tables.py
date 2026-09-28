"""Assembly of the printed comparison tables.

Ref: Table 1, pp. 6-7 - cohort and procedural characteristics by observed alert exposure
with the standardised mean difference before and after, then the two causal contrasts, the
decomposition, the alert-model performance block, the headroom row and the three-arm reader
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimators.influence import InfluenceEstimate
from ..exposure.balance import BalanceRow, balance_rows, standardised_mean_difference, worst_row
from ..protocol.schema import PersonWindowTable
from ..strata.heterogeneity import pooled_estimate
from ..utils.types import BoolArray, FloatArray

DISCRETE_CHARACTERISTICS: tuple[str, ...] = (
    "female",
    "diabetes_mellitus",
    "chronic_limb_threatening_ischaemia",
)
CONTINUOUS_CHARACTERISTICS: tuple[str, ...] = (
    "age_years",
    "lesion_length_mm",
    "operator_experience_years",
)
DISCRETE_STRATA: tuple[str, ...] = ("complexity_class", "infrapopliteal")
CONTINUOUS_STRATA: tuple[str, ...] = ("operator_experience_years",)


@dataclass(frozen=True)
class CharacteristicRow:
    """One descriptive row of the main comparison table."""

    characteristic: str
    withheld: float
    delivered: float
    smd_before: float
    smd_after: float
    kind: str

    def as_dict(self) -> dict[str, object]:
        return {
            "characteristic": self.characteristic,
            "withheld": self.withheld,
            "delivered": self.delivered,
            "smd_before": self.smd_before,
            "smd_after": self.smd_after,
            "kind": self.kind,
        }


def _share(values: FloatArray, alert: BoolArray, treated: bool) -> float:
    selected = alert if treated else ~alert
    return float(np.mean(values[selected]))


def cohort_characteristics(
    table: PersonWindowTable, weights: FloatArray | None = None
) -> list[CharacteristicRow]:
    """Characteristics with their two group summaries and both balance columns."""
    balance = {row.column: row for row in balance_rows(table, weights=weights)}
    rows: list[CharacteristicRow] = []
    for name in CONTINUOUS_CHARACTERISTICS:
        values = table.covariates[name]
        matching = balance[name]
        rows.append(
            CharacteristicRow(
                characteristic=name,
                withheld=float(np.mean(values[~table.alert])),
                delivered=float(np.mean(values[table.alert])),
                smd_before=matching.smd_before,
                smd_after=matching.smd_after,
                kind="mean_sd",
            )
        )
    for name in DISCRETE_CHARACTERISTICS:
        values = table.covariates[name]
        matching = balance[name]
        rows.append(
            CharacteristicRow(
                characteristic=name,
                withheld=_share(values, table.alert, False),
                delivered=_share(values, table.alert, True),
                smd_before=matching.smd_before,
                smd_after=matching.smd_after,
                kind="n_pct",
            )
        )
    rows.append(
        CharacteristicRow(
            characteristic="complexity_class",
            withheld=float(np.mean(table.complexity_class[~table.alert])),
            delivered=float(np.mean(table.complexity_class[table.alert])),
            smd_before=_stratum_smd(table, "complexity_class", None),
            smd_after=_stratum_smd(table, "complexity_class", weights),
            kind="distribution",
        )
    )
    rows.append(
        CharacteristicRow(
            characteristic="infrapopliteal",
            withheld=_share(table.infrapopliteal.astype(np.float64), table.alert, False),
            delivered=_share(table.infrapopliteal.astype(np.float64), table.alert, True),
            smd_before=_stratum_smd(table, "infrapopliteal", None),
            smd_after=_stratum_smd(table, "infrapopliteal", weights),
            kind="n_pct",
        )
    )
    rows.append(
        CharacteristicRow(
            characteristic="region",
            withheld=float(np.mean(table.region[~table.alert] == "Region I")),
            delivered=float(np.mean(table.region[table.alert] == "Region I")),
            smd_before=_region_smd(table, None),
            smd_after=_region_smd(table, weights),
            kind="distribution",
        )
    )
    return rows


def _stratum_smd(table: PersonWindowTable, name: str, weights: FloatArray | None) -> float:
    if name == "complexity_class":
        values = table.complexity_class.astype(np.float64)
    else:
        values = table.infrapopliteal.astype(np.float64)
    return standardised_mean_difference(values, table.alert, weights)


def _region_smd(table: PersonWindowTable, weights: FloatArray | None) -> float:
    values = (table.region == "Region I").astype(np.float64)
    return standardised_mean_difference(values, table.alert, weights)


def observed_outcomes(
    table: PersonWindowTable, mask: BoolArray | None = None
) -> dict[str, dict[str, float]]:
    """Observed outcome rates by exposure state, as the top of the outcome block."""
    selected = np.ones(table.size, dtype=np.bool_) if mask is None else mask
    subset = table.subset(selected)
    return {
        name: {
            "withheld": float(np.mean(subset.outcome(name)[~subset.alert])),
            "delivered": float(np.mean(subset.outcome(name)[subset.alert])),
            "smd": float(_binary_smd(subset.outcome(name), subset.alert)),
        }
        for name in ("takeover", "completion")
    }


def _binary_smd(values: FloatArray, alert: BoolArray) -> float:
    return standardised_mean_difference(values, alert, None)


def main_comparison_table(
    table: PersonWindowTable,
    weights: FloatArray | None,
    takeover: InfluenceEstimate,
    completion: InfluenceEstimate,
    decomposition: dict[str, dict[str, float]],
    discrimination: dict[str, object],
    reader: dict[str, object],
) -> dict[str, object]:
    """The whole main comparison table as one record."""
    rows = cohort_characteristics(table, weights)
    all_rows: list[BalanceRow] = balance_rows(table, weights=weights)
    worst_before = worst_row(all_rows, "smd_before")
    worst_after = worst_row(all_rows, "smd_after")
    return {
        "characteristics": [row.as_dict() for row in rows],
        "max_smd_before": {"column": worst_before.column, "value": worst_before.smd_before},
        "max_smd_after": {"column": worst_after.column, "value": worst_after.smd_after},
        "observed": observed_outcomes(table),
        "takeover": takeover.as_dict(),
        "completion": completion.as_dict(),
        "decomposition": decomposition,
        "alert_model": dict(discrimination),
        "reader_comparison": reader,
    }


def pooled_stratum_rows(
    groups: list[tuple[str, list[int]]],
    takeover_points: FloatArray,
    completion_points: FloatArray,
    fractions: FloatArray,
    sizes: FloatArray,
) -> list[dict[str, object]]:
    """Pooled stratum rows, size-weighted, as Table 3's last two complexity rows.

    Ref: Table 3, pp. 9-10. Each group names the strata it pools, and weighting a
    contributing stratum by its size reproduces the printed pooled values, which is why the
    same convention is used here.
    """
    columns = [
        np.asarray(values, dtype=np.float64)
        for values in (takeover_points, completion_points, fractions, sizes)
    ]
    width = columns[0].shape[0]
    if not all(column.shape[0] == width for column in columns):
        raise ValueError("the stratum columns must be aligned")
    rows: list[dict[str, object]] = []
    for label, members in groups:
        if not members:
            raise ValueError(f"pooled row {label!r} names no contributing stratum")
        if any(member < 0 or member >= width for member in members):
            raise ValueError(f"pooled row {label!r} names a stratum outside the columns")
        selected = np.asarray(members, dtype=np.int64)
        weights = columns[3][selected]
        rows.append(
            {
                "label": label,
                "takeover": pooled_estimate(columns[0][selected], weights),
                "completion": pooled_estimate(columns[1][selected], weights),
                "mediated_fraction": pooled_estimate(columns[2][selected], weights),
                "n": float(np.sum(weights)),
            }
        )
    return rows


__all__ = [
    "CONTINUOUS_CHARACTERISTICS",
    "DISCRETE_CHARACTERISTICS",
    "CharacteristicRow",
    "cohort_characteristics",
    "main_comparison_table",
    "observed_outcomes",
    "pooled_stratum_rows",
]
