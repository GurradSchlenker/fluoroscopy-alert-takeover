"""On-disk contract of the two cohort layers.

Ref: Sec. 4.5, pp. 18-19 - "clinical connection addresses the issue of linking the cine
database, navigation log, operating report, and follow-up information", and the second
layer is "the analytic person-window table".

The record-level cohort is held under site data-sharing agreements and is never
distributed, so the release ships the reader for this layout and no records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SITE_REGISTRY = "site_registry.csv"
PROCEDURES = "procedures.csv"
READER_STUDY = "reader_study.csv"
NAVIGATION_DIR = "navigation"
CINE_DIR = "cine"

SITE_REGISTRY_COLUMNS: tuple[str, ...] = ("site", "region", "annual_procedure_volume")

PROCEDURE_COLUMNS: tuple[str, ...] = (
    "procedure_id",
    "patient_id",
    "operator_id",
    "site",
    "procedure_day",
    "duration_s",
    "complexity_class",
    "infrapopliteal",
    "age_years",
    "female",
    "diabetes_mellitus",
    "chronic_limb_threatening_ischaemia",
    "lesion_length_mm",
    "reference_diameter_mm",
    "calcification_grade",
    "operator_experience_years",
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
    "preprocedure_imaging_count",
    "anaesthesia_related_event",
    "fluoroscopy_time_min",
    "contrast_volume_ml",
    "procedure_duration_min",
    "integration_cost_s",
)

LOG_COLUMNS: tuple[str, ...] = ("t_s", "kind", "score", "x_mm", "y_mm", "stage_index")

READER_COLUMNS: tuple[str, ...] = (
    "arm",
    "positive_reference",
    "negative_reference",
    "positive_calls",
    "true_positive",
    "true_negative",
)

CINE_COLUMNS: tuple[str, ...] = (
    "t_s",
    "subtraction_quality",
    "device_in_target",
    "mask_annotated",
)

LOG_KINDS: tuple[str, ...] = (
    "alert",
    "score",
    "takeover",
    "hand_back",
    "period_completed",
    "stage",
    "exchange",
    "device",
)


@dataclass(frozen=True)
class CohortLayout:
    """Resolved paths of one cohort root."""

    root: Path

    @property
    def site_registry(self) -> Path:
        return self.root / SITE_REGISTRY

    @property
    def procedures(self) -> Path:
        return self.root / PROCEDURES

    @property
    def reader_study(self) -> Path:
        return self.root / READER_STUDY

    @property
    def navigation_dir(self) -> Path:
        return self.root / NAVIGATION_DIR

    @property
    def cine_dir(self) -> Path:
        return self.root / CINE_DIR

    def navigation_log(self, procedure_id: int) -> Path:
        return self.navigation_dir / f"{procedure_id}.jsonl"

    def cine_index(self, procedure_id: int) -> Path:
        return self.cine_dir / f"{procedure_id}.csv"

    def missing(self) -> list[str]:
        """Report the contract files that are absent, so a blocked run names them."""
        absent: list[str] = []
        if not self.site_registry.is_file():
            absent.append(SITE_REGISTRY)
        if not self.procedures.is_file():
            absent.append(PROCEDURES)
        if not self.navigation_dir.is_dir():
            absent.append(NAVIGATION_DIR + "/")
        if not self.cine_dir.is_dir():
            absent.append(CINE_DIR + "/")
        return absent

    def procedure_ids(self) -> list[int]:
        """Procedure identifiers present in the navigation layer, sorted."""
        if not self.navigation_dir.is_dir():
            return []
        return sorted(int(path.stem) for path in self.navigation_dir.glob("*.jsonl"))


@dataclass
class StageTimings:
    """Wall-clock record of one pipeline stage, used by the Table A1 accounting."""

    stage: str
    seconds: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.seconds.append(float(seconds))


__all__ = [
    "CINE_COLUMNS",
    "CINE_DIR",
    "LOG_COLUMNS",
    "LOG_KINDS",
    "NAVIGATION_DIR",
    "PROCEDURES",
    "PROCEDURE_COLUMNS",
    "READER_COLUMNS",
    "READER_STUDY",
    "SITE_REGISTRY",
    "SITE_REGISTRY_COLUMNS",
    "CohortLayout",
    "StageTimings",
]
