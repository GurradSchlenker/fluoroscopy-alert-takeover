"""Assembly of the analytic person-window table from the two data layers.

Ref: Sec. 4.5, pp. 18-19 (person-window table, clinical linkage, leak prevention between
the two layers); Sec. 4.2, p. 14 (time zero and follow-up).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from ..protocol.schema import (
    CASE_MIX,
    COVARIATE_BLOCKS,
    NEGATIVE_CONTROLS,
    SAFETY_OUTCOMES,
    PersonWindowTable,
)
from ..protocol.windowing import WindowSpec, lay_windows
from ..utils.logging import get_logger
from ..utils.types import FloatArray
from .cine import CineIndex, read_cine_index
from .covariates import WindowContext, block_histories, window_covariates
from .layout import CohortLayout
from .navigation_log import NavigationLog, read_navigation_log
from .tables import ProcedureRecord, SiteRecord, load_procedures, load_sites

_LOG = get_logger("cohort.builder")


class PersonWindowRow(TypedDict):
    """One assembled row, before it is stacked into the table's arrays."""

    patient_id: int
    procedure_id: int
    operator_id: int
    site: str
    region: str
    window_index: int
    window_start_s: float
    window_length_s: float
    alert: bool
    alert_score: float
    alert_onset_s: float
    hand_back_delay_s: float
    takeover: bool
    completion: bool
    complexity_class: int
    infrapopliteal: bool
    operator_experience_years: float
    annual_procedure_volume: float
    covariates: dict[str, float]
    negative_controls: dict[str, float]
    safety: dict[str, float]
    workflow: dict[str, float]
    linkage_complete: bool


def _response_outcome(events: FloatArray, start_s: float, limit_s: float) -> bool:
    """Whether an event of this kind falls inside the response window.

    Time zero is the start of the analysis period (Sec. 4.2, p. 14); follow-up ends at
    the procedure end or the close of the response window, whichever comes first.
    """
    selected = events[(events >= start_s) & (events < limit_s)]
    return bool(selected.size > 0)


def _first_delay(events: FloatArray, start_s: float, horizon_s: float) -> float:
    """Delay from the analysis-period start to the first event before ``horizon_s``."""
    selected = events[(events >= start_s) & (events < horizon_s)]
    if selected.size == 0:
        return float("nan")
    return float(np.min(selected) - start_s)


def _procedure_rows(
    layout: CohortLayout,
    spec: WindowSpec,
    record: ProcedureRecord,
    site: SiteRecord,
    history: tuple[float, float],
) -> list[PersonWindowRow]:
    log: NavigationLog = read_navigation_log(layout.navigation_log(record.procedure_id))
    cine: CineIndex = read_cine_index(layout.cine_index(record.procedure_id))
    prior_volume, console_sessions = history
    context = WindowContext(
        record=record,
        log=log,
        cine=cine,
        annual_procedure_volume=site.annual_procedure_volume,
        operator_case_volume_prior_30d=prior_volume,
        console_sessions_this_week=console_sessions,
    )
    onsets = log.onsets()
    hand_backs = log.hand_back_times()
    takeovers = np.sort(log.t_s[np.isin(log.kind, np.asarray(["takeover"], dtype=np.str_))])
    completed = np.sort(log.t_s[np.isin(log.kind, np.asarray(["period_completed"], dtype=np.str_))])
    duration = float(record.duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"procedure {record.procedure_id} has no usable duration")

    rows: list[PersonWindowRow] = []
    for index, (start, end) in enumerate(lay_windows(duration, spec)):
        limit = min(start + spec.response_s, duration)
        inside = onsets[(onsets >= start) & (onsets < end)]
        rows.append(
            PersonWindowRow(
                patient_id=record.patient_id,
                procedure_id=record.procedure_id,
                operator_id=record.operator_id,
                site=record.site,
                region=site.region,
                window_index=index,
                window_start_s=float(start),
                window_length_s=float(end - start),
                alert=bool(inside.size > 0),
                alert_score=log.max_score(start, end),
                alert_onset_s=float(np.min(inside)) if inside.size > 0 else float("nan"),
                # The delay is recorded against the whole procedure rather than against the
                # response window, because Table 4 Panel B re-reads the mediator at widened
                # windows and needs the responses that fall beyond the pre-specified one.
                hand_back_delay_s=_first_delay(hand_backs, start, duration),
                takeover=_response_outcome(takeovers, start, limit),
                completion=_response_outcome(completed, start, limit),
                complexity_class=record.complexity_class,
                infrapopliteal=record.infrapopliteal,
                operator_experience_years=record.operator_experience_years,
                annual_procedure_volume=site.annual_procedure_volume,
                covariates=window_covariates(context, start, end),
                negative_controls={
                    name: float(record.outcomes[name]) for name in NEGATIVE_CONTROLS
                },
                safety={
                    name: float(record.outcomes[name]) if name in record.outcomes else float("nan")
                    for name in SAFETY_OUTCOMES
                },
                workflow=dict(record.workflow),
                linkage_complete=bool(cine.t_s.size > 0 and np.isfinite(duration)),
            )
        )
    return rows


def _column(rows: list[PersonWindowRow], name: str, dtype: type) -> np.ndarray:
    """Stack one scalar field of the rows.

    The field is named by a variable, so the row is read as a plain mapping; the schema's
    own field names are checked when the row is constructed rather than here.
    """
    if name not in PersonWindowRow.__annotations__:
        raise KeyError(f"unknown person-window field: {name}")
    return np.asarray([dict(row)[name] for row in rows], dtype=dtype)


def _assemble(rows: list[PersonWindowRow], spec: WindowSpec) -> PersonWindowTable:
    """Turn the row mappings into the immutable table."""
    if not rows:
        raise ValueError("no person-windows were produced; check the window specification")
    members = [name for block in COVARIATE_BLOCKS.values() for name in block] + list(CASE_MIX)
    delays = _column(rows, "hand_back_delay_s", np.float64)
    return PersonWindowTable(
        patient_id=_column(rows, "patient_id", np.int64),
        procedure_id=_column(rows, "procedure_id", np.int64),
        operator_id=_column(rows, "operator_id", np.int64),
        site=_column(rows, "site", np.str_),
        region=_column(rows, "region", np.str_),
        window_index=_column(rows, "window_index", np.int64),
        window_start_s=_column(rows, "window_start_s", np.float64),
        window_length_s=_column(rows, "window_length_s", np.float64),
        alert=_column(rows, "alert", np.bool_),
        alert_score=_column(rows, "alert_score", np.float64),
        alert_onset_s=_column(rows, "alert_onset_s", np.float64),
        hand_back_delay_s=delays,
        base_window_s=spec.response_s,
        takeover=_column(rows, "takeover", np.bool_),
        completion=_column(rows, "completion", np.bool_),
        hand_back=np.isfinite(delays) & (delays <= spec.response_s),
        complexity_class=_column(rows, "complexity_class", np.int64),
        infrapopliteal=_column(rows, "infrapopliteal", np.bool_),
        operator_experience_years=_column(rows, "operator_experience_years", np.float64),
        annual_procedure_volume=_column(rows, "annual_procedure_volume", np.float64),
        covariates={
            name: np.asarray([row["covariates"][name] for row in rows], dtype=np.float64)
            for name in members
        },
        negative_controls={
            name: np.asarray([row["negative_controls"][name] for row in rows], dtype=np.float64)
            for name in NEGATIVE_CONTROLS
        },
        safety={
            name: np.asarray([row["safety"][name] for row in rows], dtype=np.float64)
            for name in SAFETY_OUTCOMES
        },
        workflow={
            metric: np.asarray([row["workflow"][metric] for row in rows], dtype=np.float64)
            for metric in rows[0]["workflow"]
        },
        linkage_complete=_column(rows, "linkage_complete", np.bool_),
    )


def build_person_windows(
    layout: CohortLayout,
    spec: WindowSpec,
    procedure_subset: list[int] | None = None,
) -> PersonWindowTable:
    """Read both layers and return the person-window table.

    Procedures are processed one at a time so the peak memory of the build is one
    procedure's log and cine index.
    """
    absent = layout.missing()
    if absent:
        raise FileNotFoundError(
            "cohort root does not satisfy the release layout; missing: " + ", ".join(absent)
        )
    sites = load_sites(layout.site_registry)
    records = load_procedures(layout.procedures)
    if procedure_subset is not None:
        wanted = set(procedure_subset)
        records = [record for record in records if record.procedure_id in wanted]
    history = block_histories(records)

    rows: list[PersonWindowRow] = []
    for record in records:
        site = sites.get(record.site)
        if not isinstance(site, SiteRecord):
            raise KeyError(f"site absent from the registry: {record.site}")
        rows.extend(_procedure_rows(layout, spec, record, site, history[record.procedure_id]))

    table = _assemble(rows, spec)
    _LOG.info(
        "person-window table built: %d rows over %d procedures and %d sites",
        table.size,
        len({record.procedure_id for record in records}),
        table.n_sites,
    )
    return table


def cohort_digest(table: PersonWindowTable) -> dict[str, float]:
    """Descriptive digest of a built table, written next to the artefacts."""
    return {
        "rows": float(table.size),
        "patients": float(table.n_patients),
        "sites": float(table.n_sites),
        "alert_rate": float(np.mean(table.alert)),
        "takeover_rate": float(np.mean(table.takeover)),
        "completion_rate": float(np.mean(table.completion)),
        "hand_back_rate": float(np.mean(table.hand_back)),
    }


__all__ = ["PersonWindowRow", "build_person_windows", "cohort_digest"]
