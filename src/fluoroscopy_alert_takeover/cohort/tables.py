"""Tabular cohort files: the site registry and the operating-report layer.

Ref: Sec. 4.5, pp. 18-19 - inclusion and exclusion criteria, enrolment, outcome
verification and clinical linkage all sit on the operating-report layer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout import PROCEDURE_COLUMNS, SITE_REGISTRY_COLUMNS

_NUMERIC_PROCEDURE_COLUMNS = frozenset(PROCEDURE_COLUMNS) - {
    "site",
}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file with a header row into a list of mappings."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"cohort table not found: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"cohort table has no header: {resolved}")
        return [dict(row) for row in reader]


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return float(raw)


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return int(float(raw))


def _as_flag(row: dict[str, str], key: str, default: bool = False) -> bool:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class SiteRecord:
    """One participating site and its region."""

    site: str
    region: str
    annual_procedure_volume: float


@dataclass(frozen=True)
class ProcedureRecord:
    """One procedure: case mix, operating-report outcomes and workflow quantities."""

    procedure_id: int
    patient_id: int
    operator_id: int
    site: str
    procedure_day: int
    duration_s: float
    complexity_class: int
    infrapopliteal: bool
    age_years: float
    female: bool
    diabetes_mellitus: bool
    chronic_limb_threatening_ischaemia: bool
    lesion_length_mm: float
    reference_diameter_mm: float
    calcification_grade: float
    operator_experience_years: float
    outcomes: dict[str, bool]
    workflow: dict[str, float]


_OUTCOME_KEYS = (
    "technical_success",
    "treatment_success",
    "limb_event",
    "major_adverse_cardiac_event",
    "death",
    "amputation",
    "transfusion_or_surgery",
    "emergency_representation_7d",
    "unrelated_diagnosis_admission",
    "contralateral_limb_intervention",
    "anaesthesia_related_event",
)

_WORKFLOW_KEYS = (
    "fluoroscopy_time_min",
    "contrast_volume_ml",
    "procedure_duration_min",
    "integration_cost_s",
)


def load_sites(path: str | Path) -> dict[str, SiteRecord]:
    """Read the site registry keyed by site identifier."""
    records: dict[str, SiteRecord] = {}
    for row in read_csv_rows(path):
        missing = [column for column in SITE_REGISTRY_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"site registry missing columns: {missing}")
        site = row["site"].strip()
        records[site] = SiteRecord(
            site=site,
            region=row["region"].strip(),
            annual_procedure_volume=float(row["annual_procedure_volume"]),
        )
    return records


def load_procedures(path: str | Path) -> list[ProcedureRecord]:
    """Read the operating-report layer."""
    records: list[ProcedureRecord] = []
    for row in read_csv_rows(path):
        absent = [column for column in PROCEDURE_COLUMNS if column not in row]
        if absent:
            raise ValueError(f"procedure table missing columns: {absent}")
        outcomes = {key: _as_flag(row, key) for key in _OUTCOME_KEYS}
        # ``completion`` is a property of the analysis period, not of the procedure, so it is
        # read from the navigation log rather than from the operating-report layer (Sec. 4.2,
        # p. 14: the outcome is assessed at the end of the follow-up of each period).
        # Table 4 Panel A reports this negative control in percentage points alongside the
        # binary ones, so the operating-report layer carries it binarised at the site
        # median (release schema; the manuscript prints the row, not its scale).
        outcomes["preprocedure_imaging_count"] = _as_flag(row, "preprocedure_imaging_count")
        records.append(
            ProcedureRecord(
                procedure_id=_as_int(row, "procedure_id"),
                patient_id=_as_int(row, "patient_id"),
                operator_id=_as_int(row, "operator_id"),
                site=row["site"].strip(),
                procedure_day=_as_int(row, "procedure_day"),
                duration_s=_as_float(row, "duration_s"),
                complexity_class=_as_int(row, "complexity_class", 1),
                infrapopliteal=_as_flag(row, "infrapopliteal"),
                age_years=_as_float(row, "age_years"),
                female=_as_flag(row, "female"),
                diabetes_mellitus=_as_flag(row, "diabetes_mellitus"),
                chronic_limb_threatening_ischaemia=_as_flag(
                    row, "chronic_limb_threatening_ischaemia"
                ),
                lesion_length_mm=_as_float(row, "lesion_length_mm"),
                reference_diameter_mm=_as_float(row, "reference_diameter_mm"),
                calcification_grade=_as_float(row, "calcification_grade"),
                operator_experience_years=_as_float(row, "operator_experience_years"),
                outcomes=outcomes,
                workflow={key: _as_float(row, key) for key in _WORKFLOW_KEYS},
            )
        )
    if not records:
        raise ValueError(f"procedure table is empty: {path}")
    return records


def registry_value(registry: dict[str, SiteRecord], site: str, attribute: str) -> Any:
    """Read one attribute of a site, failing loudly when the site is unknown."""
    if site not in registry:
        raise KeyError(f"site absent from the registry: {site}")
    return getattr(registry[site], attribute)


__all__ = [
    "ProcedureRecord",
    "SiteRecord",
    "load_procedures",
    "load_sites",
    "read_csv_rows",
    "registry_value",
]
