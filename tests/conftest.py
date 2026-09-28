"""Test data for the suite.

The release ships no data-producing code. The cohort this analysis was written for is a
restricted clinical set and nothing is put in its place: the numbers below are typed out,
with no mechanism, coefficients or distribution behind them. A test that needs a
person-window table states the rows it needs, and a test that needs the reader writes those
same numbers to disk first.

The numbers are chosen so the suite has something to work on - both exposure states in every
procedure, three complexity classes, three sites, three experience bands, three volume
tertiles, one infrapopliteal procedure, one response that lands beyond the response window
and one hand-back inside an unexposed window - and for no other reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fluoroscopy_alert_takeover.protocol.schema import (
    CASE_MIX,
    COVARIATE_BLOCKS,
    NEGATIVE_CONTROLS,
    SAFETY_OUTCOMES,
    PersonWindowTable,
)
from fluoroscopy_alert_takeover.protocol.windowing import WindowSpec
from fluoroscopy_alert_takeover.utils.types import BoolArray, FloatArray

WINDOW_LENGTH_S = 30.0
RESPONSE_WINDOW_S = 30.0
WINDOWS_PER_PROCEDURE = 6
FRAMES_PER_WINDOW = 4

# Column order of every row below.
COLUMNS = (
    "procedure_id",
    "window_index",
    "alert",
    "alert_score",
    "hand_back_delay_s",
    "takeover",
    "completion",
    "calcification_grade",
    "lesion_length_mm",
    "subtraction_quality",
    "advance_rate_mm_per_s",
    "cine_frames_prior",
    "age_years",
    "female",
    "emergency_representation_7d",
    "unrelated_diagnosis_admission",
    "limb_event",
    "technical_success",
    "fluoroscopy_time_min",
    "contrast_volume_ml",
)

# One tuple per person-window, in the column order above. A delay of -1 means no hand-back
# was logged for that window.
ROWS: tuple[tuple[float, ...], ...] = (
    (
        1,
        0,
        1,
        0.62,
        4.0,
        1,
        1,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        1,
        1,
        0,
        0.31,
        -1.0,
        0,
        1,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        1,
        2,
        1,
        0.58,
        -1.0,
        1,
        1,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        1,
        3,
        0,
        0.22,
        18.0,
        0,
        1,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        1,
        4,
        0,
        0.19,
        -1.0,
        0,
        0,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        1,
        5,
        1,
        0.55,
        -1.0,
        0,
        0,
        0.8,
        90.0,
        0.8,
        5.5,
        30.0,
        64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        19.2,
        84.7,
    ),
    (
        2,
        0,
        0,
        0.24,
        -1.0,
        0,
        0,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        2,
        1,
        1,
        0.55,
        7.0,
        1,
        1,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        2,
        2,
        0,
        0.29,
        -1.0,
        0,
        0,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        2,
        3,
        0,
        0.17,
        6.0,
        1,
        1,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        2,
        4,
        1,
        0.61,
        3.0,
        1,
        1,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        2,
        5,
        0,
        0.26,
        -1.0,
        0,
        0,
        1.1,
        110.0,
        0.78,
        6.1,
        34.0,
        59.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        17.4,
        71.3,
    ),
    (
        3,
        0,
        1,
        0.57,
        5.0,
        1,
        1,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        3,
        1,
        1,
        0.63,
        -1.0,
        0,
        0,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        3,
        2,
        0,
        0.33,
        -1.0,
        0,
        1,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        3,
        3,
        1,
        0.52,
        11.0,
        1,
        1,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        3,
        4,
        0,
        0.28,
        -1.0,
        0,
        0,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        3,
        5,
        0,
        0.21,
        12.0,
        0,
        1,
        1.6,
        145.0,
        0.75,
        6.4,
        40.0,
        71.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        21.6,
        96.1,
    ),
    (
        4,
        0,
        0,
        0.3,
        -1.0,
        0,
        0,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        4,
        1,
        1,
        0.66,
        6.0,
        1,
        1,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        4,
        2,
        1,
        0.59,
        14.0,
        1,
        1,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        4,
        3,
        0,
        0.27,
        26.0,
        0,
        1,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        4,
        4,
        1,
        0.64,
        -1.0,
        0,
        0,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        4,
        5,
        1,
        0.58,
        8.0,
        1,
        1,
        2.1,
        160.0,
        0.72,
        6.8,
        44.0,
        76.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        23.8,
        103.5,
    ),
    (
        5,
        0,
        1,
        0.6,
        3.0,
        1,
        1,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        5,
        1,
        0,
        0.35,
        5.0,
        0,
        1,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        5,
        2,
        1,
        0.55,
        34.0,
        1,
        0,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        5,
        3,
        1,
        0.61,
        -1.0,
        0,
        0,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        5,
        4,
        0,
        0.24,
        21.0,
        0,
        1,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        5,
        5,
        0,
        0.16,
        -1.0,
        0,
        0,
        2.6,
        195.0,
        0.7,
        7.2,
        52.0,
        81.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        26.1,
        118.4,
    ),
    (
        6,
        0,
        0,
        0.22,
        -1.0,
        0,
        0,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
    (
        6,
        1,
        0,
        0.18,
        -1.0,
        0,
        0,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
    (
        6,
        2,
        1,
        0.68,
        12.0,
        1,
        1,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
    (
        6,
        3,
        0,
        0.31,
        -1.0,
        0,
        0,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
    (
        6,
        4,
        1,
        0.63,
        -1.0,
        0,
        0,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
    (
        6,
        5,
        1,
        0.66,
        1.0,
        1,
        1,
        3.1,
        210.0,
        0.66,
        7.6,
        58.0,
        68.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        28.3,
        132.9,
    ),
)

# Per-procedure descriptions, keyed by procedure_id. These are the numbers the tests vary
# across procedures: the site and region, the complexity class, whether the disease is
# infrapopliteal, the operator's experience, the site's annual volume, and four covariates
# that differ from procedure to procedure.
PROCEDURES: dict[int, dict[str, float | str | bool]] = {
    1: {
        "site": "Site A",
        "region": "Region I",
        "complexity": 1,
        "infra": False,
        "experience": 2.0,
        "volume": 640.0,
        "diameter": 3.9,
        "curvature": 0.31,
        "reversals": 3.0,
        "diabetes": 0.0,
        "clti": 0.0,
    },
    2: {
        "site": "Site B",
        "region": "Region I",
        "complexity": 1,
        "infra": False,
        "experience": 2.5,
        "volume": 520.0,
        "diameter": 3.7,
        "curvature": 0.34,
        "reversals": 4.0,
        "diabetes": 1.0,
        "clti": 0.0,
    },
    3: {
        "site": "Site C",
        "region": "Region II",
        "complexity": 2,
        "infra": False,
        "experience": 5.0,
        "volume": 380.0,
        "diameter": 3.6,
        "curvature": 0.38,
        "reversals": 5.0,
        "diabetes": 0.0,
        "clti": 1.0,
    },
    4: {
        "site": "Site A",
        "region": "Region I",
        "complexity": 2,
        "infra": False,
        "experience": 6.0,
        "volume": 640.0,
        "diameter": 3.4,
        "curvature": 0.42,
        "reversals": 6.0,
        "diabetes": 1.0,
        "clti": 1.0,
    },
    5: {
        "site": "Site B",
        "region": "Region I",
        "complexity": 3,
        "infra": False,
        "experience": 12.0,
        "volume": 520.0,
        "diameter": 3.1,
        "curvature": 0.47,
        "reversals": 7.0,
        "diabetes": 0.0,
        "clti": 1.0,
    },
    6: {
        "site": "Site C",
        "region": "Region II",
        "complexity": 3,
        "infra": True,
        "experience": 14.0,
        "volume": 380.0,
        "diameter": 2.8,
        "curvature": 0.55,
        "reversals": 9.0,
        "diabetes": 1.0,
        "clti": 1.0,
    },
}

# Columns the tests do not vary. Each is pinned to one number, so a test that reads it reads
# something stated rather than something derived.
CONSTANTS: dict[str, float] = {
    "run_duration_s": 24.0,
    "dwell_time_s": 1.6,
    "guidewire_exchanges": 1.0,
    "cine_runs_prior": 2.0,
    "prior_device_advance_mm": 60.0,
    "operator_case_volume_prior_30d": 12.0,
    "console_sessions_this_week": 3.0,
    "contralateral_limb_intervention": 0.0,
    "preprocedure_imaging_count": 0.0,
    "anaesthesia_related_event": 0.0,
    "major_adverse_cardiac_event": 0.0,
    "death": 0.0,
    "amputation": 0.0,
    "transfusion_or_surgery": 0.0,
    "procedure_duration_min": 78.4,
    "integration_cost_s": 4.2,
}

# The two frames and two masks the cine layer alternates between. Every value is written out.
FRAME_A: tuple[tuple[float, ...], ...] = (
    (0.10, 0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10),
    (0.10, 0.20, 0.40, 0.60, 0.70, 0.50, 0.30, 0.10),
    (0.20, 0.40, 0.70, 0.90, 0.90, 0.70, 0.40, 0.20),
    (0.30, 0.60, 0.90, 1.00, 1.00, 0.90, 0.60, 0.30),
    (0.20, 0.50, 0.80, 0.95, 0.95, 0.80, 0.50, 0.20),
    (0.10, 0.30, 0.50, 0.70, 0.70, 0.50, 0.30, 0.10),
    (0.10, 0.20, 0.30, 0.40, 0.40, 0.30, 0.20, 0.10),
    (0.00, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10, 0.00),
)
FRAME_B: tuple[tuple[float, ...], ...] = (
    (0.00, 0.00, 0.10, 0.10, 0.20, 0.40, 0.60, 0.70),
    (0.00, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.90),
    (0.10, 0.20, 0.30, 0.50, 0.60, 0.80, 0.90, 0.95),
    (0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.90),
    (0.10, 0.20, 0.40, 0.60, 0.70, 0.80, 0.85, 0.80),
    (0.00, 0.10, 0.20, 0.40, 0.50, 0.60, 0.65, 0.60),
    (0.00, 0.00, 0.10, 0.20, 0.30, 0.40, 0.45, 0.40),
    (0.00, 0.00, 0.00, 0.10, 0.10, 0.20, 0.25, 0.20),
)
MASK_A: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 1, 1, 0),
    (0, 0, 1, 1, 1, 1, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)
MASK_B: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 1, 1, 1),
    (0, 0, 0, 1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# The three reader arms, as counts against the reference standard.
READER_ARMS: tuple[tuple[str, int, int, int, int, int], ...] = (
    ("clinician_alone", 300, 900, 200, 184, 831),
    ("alert_alone", 300, 900, 300, 254, 793),
    ("team", 300, 900, 290, 252, 812),
)

PROCEDURE_HEADER = (
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


def _column(rows: tuple[tuple[float, ...], ...], name: str) -> FloatArray:
    position = COLUMNS.index(name)
    return np.asarray([row[position] for row in rows], dtype=np.float64)


def build_table() -> PersonWindowTable:
    """Assemble the person-window table the suite reads."""
    rows = ROWS
    procedure = _column(rows, "procedure_id")
    window = _column(rows, "window_index")
    delay = _column(rows, "hand_back_delay_s")
    onset = np.where(_column(rows, "alert") > 0.5, window * WINDOW_LENGTH_S + 1.0, np.nan)

    def per_procedure(name: str) -> FloatArray:
        return np.asarray([float(PROCEDURES[int(p)][name]) for p in procedure], dtype=np.float64)

    covariates: dict[str, FloatArray] = {
        "calcification_grade": _column(rows, "calcification_grade"),
        "lesion_length_mm": _column(rows, "lesion_length_mm"),
        "reference_diameter_mm": per_procedure("diameter"),
        "cine_frames_prior": _column(rows, "cine_frames_prior"),
        "subtraction_quality": _column(rows, "subtraction_quality"),
        "advance_rate_mm_per_s": _column(rows, "advance_rate_mm_per_s"),
        "tip_path_curvature": per_procedure("curvature"),
        "motion_reversals": per_procedure("reversals"),
        "run_duration_s": np.full(rows_len := len(rows), CONSTANTS["run_duration_s"]),
        "dwell_time_s": np.full(rows_len, CONSTANTS["dwell_time_s"]),
        "elapsed_procedure_time_s": window * WINDOW_LENGTH_S,
        "stage_index": window + 1.0,
        "guidewire_exchanges": np.full(rows_len, CONSTANTS["guidewire_exchanges"]),
        "cine_runs_prior": np.full(rows_len, CONSTANTS["cine_runs_prior"]),
        "prior_device_advance_mm": np.full(rows_len, CONSTANTS["prior_device_advance_mm"]),
        "operator_experience_years": per_procedure("experience"),
        "annual_procedure_volume": per_procedure("volume"),
        "operator_case_volume_prior_30d": np.full(
            rows_len, CONSTANTS["operator_case_volume_prior_30d"]
        ),
        "console_sessions_this_week": np.full(rows_len, CONSTANTS["console_sessions_this_week"]),
        "age_years": _column(rows, "age_years"),
        "female": _column(rows, "female"),
        "diabetes_mellitus": per_procedure("diabetes"),
        "chronic_limb_threatening_ischaemia": per_procedure("clti"),
    }
    members = [name for block in COVARIATE_BLOCKS.values() for name in block] + list(CASE_MIX)
    missing = [name for name in members if name not in covariates]
    if missing:
        raise AssertionError(f"the test table leaves columns unset: {missing}")

    negative = {name: np.zeros(rows_len) for name in NEGATIVE_CONTROLS}
    negative["emergency_representation_7d"] = _column(rows, "emergency_representation_7d")
    negative["unrelated_diagnosis_admission"] = _column(rows, "unrelated_diagnosis_admission")

    safety = {name: np.zeros(rows_len) for name in SAFETY_OUTCOMES}
    safety["technical_success"] = _column(rows, "technical_success")
    safety["treatment_success"] = _column(rows, "technical_success")
    safety["limb_event"] = _column(rows, "limb_event")

    return PersonWindowTable(
        patient_id=procedure.astype(np.int64),
        procedure_id=procedure.astype(np.int64),
        operator_id=procedure.astype(np.int64),
        site=np.asarray([str(PROCEDURES[int(p)]["site"]) for p in procedure], dtype=np.str_),
        region=np.asarray([str(PROCEDURES[int(p)]["region"]) for p in procedure], dtype=np.str_),
        window_index=window.astype(np.int64),
        window_start_s=window * WINDOW_LENGTH_S,
        window_length_s=np.full(rows_len, WINDOW_LENGTH_S),
        alert=_column(rows, "alert") > 0.5,
        alert_score=_column(rows, "alert_score"),
        alert_onset_s=onset,
        hand_back_delay_s=delay,
        base_window_s=RESPONSE_WINDOW_S,
        takeover=_column(rows, "takeover") > 0.5,
        completion=_column(rows, "completion") > 0.5,
        hand_back=(delay >= 0.0) & (delay <= RESPONSE_WINDOW_S),
        complexity_class=np.asarray(
            [int(PROCEDURES[int(p)]["complexity"]) for p in procedure], dtype=np.int64
        ),
        infrapopliteal=np.asarray(
            [bool(PROCEDURES[int(p)]["infra"]) for p in procedure], dtype=np.bool_
        ),
        operator_experience_years=covariates["operator_experience_years"],
        annual_procedure_volume=covariates["annual_procedure_volume"],
        covariates=covariates,
        negative_controls=negative,
        safety=safety,
        workflow={
            "fluoroscopy_time_min": _column(rows, "fluoroscopy_time_min"),
            "contrast_volume_ml": _column(rows, "contrast_volume_ml"),
            "procedure_duration_min": np.full(rows_len, CONSTANTS["procedure_duration_min"]),
            "integration_cost_s": np.full(rows_len, CONSTANTS["integration_cost_s"]),
        },
        linkage_complete=np.ones(rows_len, dtype=np.bool_),
    )


def window_spec() -> WindowSpec:
    """The analysis-period layout the sample below is laid out on."""
    return WindowSpec(
        length_s=WINDOW_LENGTH_S, stride_s=WINDOW_LENGTH_S, response_s=RESPONSE_WINDOW_S
    )


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def write_sample(root: Path) -> Path:
    """Write the rows above in the on-disk contract of ``cohort.layout``.

    The cohort reader and the alert-policy dataset are the only places that read files
    rather than arrays, so the suite gives them exactly the numbers above.
    """
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("navigation", "cine", "frames", "masks"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    numbers = sorted(PROCEDURES)
    by_procedure: dict[int, list[tuple[float, ...]]] = {number: [] for number in numbers}
    for row in ROWS:
        by_procedure[int(row[0])].append(row)

    registry: list[list[object]] = []
    seen: set[str] = set()
    for number in numbers:
        site = str(PROCEDURES[number]["site"])
        if site in seen:
            continue
        seen.add(site)
        registry.append([site, PROCEDURES[number]["region"], PROCEDURES[number]["volume"]])
    _write_csv(root / "site_registry.csv", ("site", "region", "annual_procedure_volume"), registry)

    report: list[list[object]] = []
    for number in numbers:
        first = by_procedure[number][0]
        info = PROCEDURES[number]
        report.append(
            [
                number,
                number,
                number,
                info["site"],
                0,
                WINDOWS_PER_PROCEDURE * WINDOW_LENGTH_S,
                info["complexity"],
                int(bool(info["infra"])),
                first[12],
                int(first[13]),
                int(info["diabetes"]),
                int(info["clti"]),
                first[8],
                info["diameter"],
                first[7],
                info["experience"],
                int(first[17]),
                int(first[17]),
                int(first[16]),
                0,
                0,
                0,
                0,
                int(first[14]),
                int(first[15]),
                0,
                0,
                0,
                first[18],
                first[19],
                CONSTANTS["procedure_duration_min"],
                CONSTANTS["integration_cost_s"],
            ]
        )
    _write_csv(root / "procedures.csv", PROCEDURE_HEADER, report)

    frame_a = np.asarray(FRAME_A, dtype=np.float32)
    frame_b = np.asarray(FRAME_B, dtype=np.float32)
    mask_a = np.asarray(MASK_A, dtype=np.uint8)
    mask_b = np.asarray(MASK_B, dtype=np.uint8)
    for number in numbers:
        cine: list[list[object]] = []
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        events: list[dict[str, object]] = []
        for row in by_procedure[number]:
            start = row[1] * WINDOW_LENGTH_S
            events.append({"t_s": start, "kind": "score", "score": row[3]})
            events.append({"t_s": start + 0.5, "kind": "stage", "stage_index": int(row[1]) + 1})
            events.append({"t_s": start + 0.6, "kind": "exchange"})
            if row[2] > 0.5:
                events.append({"t_s": start + 1.0, "kind": "alert", "score": row[3]})
            if row[4] >= 0.0:
                events.append({"t_s": start + row[4], "kind": "hand_back"})
            if row[5] > 0.5:
                events.append({"t_s": start + 3.0, "kind": "takeover"})
            if row[6] > 0.5:
                events.append({"t_s": start + 2.0, "kind": "period_completed"})
            for step in range(FRAMES_PER_WINDOW):
                at = start + step * 7.5
                events.append(
                    {"t_s": at, "kind": "device", "x_mm": 20.0 + step, "y_mm": 10.0 + row[1]}
                )
                cine.append([at, row[9], int(row[2] > 0.5), int(step % 2 == 0)])
                frames.append(frame_a if step % 2 == 0 else frame_b)
                masks.append(mask_a if step % 2 == 0 else mask_b)
        _write_csv(
            root / "cine" / f"{number}.csv",
            ("t_s", "subtraction_quality", "device_in_target", "mask_annotated"),
            cine,
        )
        with (root / "navigation" / f"{number}.jsonl").open("w", encoding="utf-8") as handle:
            for event in sorted(events, key=lambda item: float(item["t_s"])):  # type: ignore[arg-type]
                handle.write(json.dumps(event) + "\n")
        np.save(root / "frames" / f"{number}.npy", np.stack(frames))
        np.save(root / "masks" / f"{number}.npy", np.stack(masks))

    _write_csv(
        root / "reader_study.csv",
        (
            "arm",
            "positive_reference",
            "negative_reference",
            "positive_calls",
            "true_positive",
            "true_negative",
        ),
        [list(arm) for arm in READER_ARMS],
    )
    return root


@pytest.fixture(scope="session")
def windows() -> PersonWindowTable:
    """The person-window table the unit tests read."""
    return build_table()


@pytest.fixture(scope="session")
def sample_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same numbers written in the on-disk contract, for the reader tests."""
    return write_sample(tmp_path_factory.mktemp("sample"))


@pytest.fixture(scope="session")
def alerts(windows: PersonWindowTable) -> BoolArray:
    """The exposure column of the test table."""
    return windows.alert
