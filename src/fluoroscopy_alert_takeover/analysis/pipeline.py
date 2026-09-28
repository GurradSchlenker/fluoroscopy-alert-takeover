"""The end-to-end analysis pipeline.

Ref: Sec. 4.2-4.10, pp. 14-21, in the order the manuscript states them: emulate the trial
on the person-window table; probe the identification assumptions; build the
certified-overlap gate; estimate the two causal contrasts with the cross-fitted one-step
estimator of Eq. (2); decompose the completion effect under the interventional analogue of
Eq. (1); run the pre-specified falsification pair; run the estimator and adjustment-set
ablation; break the effect down by stratum; and select the operating point over the
abstention band.

Every value the manuscript does not print is a field of :class:`AnalysisConfig` carrying a
documented engineering default; the quantitites the manuscript does print are protocol
constants and keep their printed values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..estimands.attenuation import AttenuationPoint, window_attenuation
from ..estimands.interventional import (
    PathEstimates,
    augmented_psi_terms,
    mediated_fraction,
    theta_null_test,
)
from ..estimators.ablation import (
    AblationRow,
    ablation_deficit_price,
    estimator_spread,
    nuisance_drift,
    run_ablation_grid,
)
from ..estimators.aipw import aipw_contrast
from ..estimators.crossfit import TrimResult, clip_propensity, cross_fitted_nuisance, table_design
from ..estimators.decomposition import (
    MediationNuisance,
    cross_fitted_decomposition,
    cross_fitted_mediation_nuisance,
)
from ..estimators.influence import InfluenceEstimate
from ..estimators.nuisance import CrossFittedNuisance, cross_fitted_outcome
from ..evaluation.reader_study import ArmCounts, reader_comparison
from ..exposure.model import ExposureModel, fit_balance_first_exposure
from ..falsification.protocol import FalsificationReport, run_falsification
from ..forest.causal_forest import ForestConfig, HonestSplitCausalForest, forest_strata
from ..forest.gradient import GradientTest, monotone_gradient_test
from ..inference.alternatives import (
    bootstrap_row,
    cluster_robust_rows,
    leave_one_site_out,
    solve_shift_for_marginal_rate,
    stochastic_intervention_contrast,
)
from ..multiplicity.corrections import holm_step_down
from ..operating_point.abstention import (
    AbstentionBand,
    OperatingPoint,
    abstention_band_comparison,
    select_operating_point,
)
from ..operating_point.benefit_curve import BenefitCurve, causal_benefit_curve
from ..overlap.gate import (
    DEFAULT_GRID,
    DEFAULT_TOLERANCE,
    CertifiedGate,
    build_certified_overlap_gate,
)
from ..protocol.folds import site_disjoint_patient_folds
from ..protocol.probes import (
    ProbeResult,
    consistency_probe,
    exchangeability_probe,
    positivity_probe,
)
from ..protocol.schema import NEGATIVE_CONTROLS, PersonWindowTable
from ..reporting.success import proportion_of_flag, success_criteria
from ..reporting.tables import main_comparison_table, pooled_stratum_rows
from ..strata.gap import maximum_subgroup_gap
from ..strata.partitions import (
    complexity_partition,
    experience_partition,
    pooled_complexity_partition,
    silent_override_rate,
    site_partition,
    vessel_bed_partition,
    volume_tertile_partition,
)
from ..utils.logging import get_logger
from ..utils.types import BoolArray, FloatArray, IntArray
from ..workflow.procedure_metrics import describe, integration_cost

_LOG = get_logger("analysis.pipeline")

ContrastPair = Callable[[PersonWindowTable], tuple[InfluenceEstimate, InfluenceEstimate]]


@dataclass(frozen=True)
class AnalysisConfig:
    """Protocol constants and engineering defaults of one analysis run."""

    n_folds: int = 5
    clip_gamma: float = 0.01
    outcome_learner: str = "glm"
    gate_tolerance: float = DEFAULT_TOLERANCE
    gate_grid: tuple[float, ...] = DEFAULT_GRID
    seed: int = 20260101
    n_bootstrap: int = 1000
    bootstrap_seed: int = 20260102
    primary_outcome: str = "takeover"
    secondary_outcome: str = "completion"
    mediator: str = "hand_back"
    consistency_floor: float = 0.90
    gate_ess_floor: float = 0.20
    alpha: float = 0.05
    abstention: AbstentionBand = field(default_factory=lambda: AbstentionBand(0.30, 0.40))
    thresholds: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.55)
    window_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    forest: ForestConfig = field(default_factory=ForestConfig)

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("cross-fitting needs at least two folds")
        if not 0.0 < self.clip_gamma < 0.5:
            raise ValueError("the trimming bound must lie in (0, 0.5)")


@dataclass(frozen=True)
class StratumRow:
    """One row of the per-condition breakdown."""

    label: str
    group: str
    n: int
    takeover: float
    takeover_se: float
    takeover_p: float
    completion: float
    completion_se: float
    completion_p: float
    mediated_fraction: float
    mediated_fraction_se: float
    silent_override: float

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "group": self.group,
            "n": float(self.n),
            "takeover_pp": self.takeover,
            "takeover_se_pp": self.takeover_se,
            "takeover_p": self.takeover_p,
            "completion_pp": self.completion,
            "completion_se_pp": self.completion_se,
            "completion_p": self.completion_p,
            "mediated_fraction": self.mediated_fraction,
            "mediated_fraction_se": self.mediated_fraction_se,
            "silent_override": self.silent_override,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Every quantity the release reports for one analysis run."""

    config: AnalysisConfig
    exposure: ExposureModel
    gate: CertifiedGate
    trimmed: TrimResult
    probes: list[ProbeResult]
    primary: InfluenceEstimate
    secondary: InfluenceEstimate
    paths: PathEstimates
    theta: InfluenceEstimate
    mediator_diagnostics: dict[str, float]
    theta_test: dict[str, float]
    falsification: FalsificationReport
    ablation: list[AblationRow]
    ablation_summary: dict[str, float]
    strata: list[StratumRow]
    pooled: list[dict[str, object]]
    gradient: GradientTest
    subgroup_gap: dict[str, object]
    alternatives: dict[str, object]
    attenuation: list[AttenuationPoint]
    benefit_curve: BenefitCurve
    operating_point: OperatingPoint
    band_comparison: dict[str, object]
    success: dict[str, object]
    table_record: dict[str, object]
    family_multiplicity: dict[str, object]
    reader: dict[str, object]

    def headline(self) -> dict[str, object]:
        """The two contrasts, the decomposition and the frozen operating point."""
        return {
            "takeover": self.primary.as_dict(),
            "completion": self.secondary.as_dict(),
            "decomposition": self.paths.as_dict(),
            "mediated_fraction": self.theta.as_dict(),
            "mediated_fraction_test": self.theta_test,
            "operating_point": self.operating_point.as_dict(),
            "ablation_summary": self.ablation_summary,
            "gate": {"epsilon": self.gate.epsilon, "retained_share": self.gate.retained_share},
            "probes": [probe.__dict__ for probe in self.probes],
        }


def contrast_pair(config: AnalysisConfig) -> ContrastPair:
    """Return the pre-specified analysis as a function of a person-window table.

    The leave-one-site-out validation, the causal benefit curve and the operating-point
    sweep all consume this callable, so those stages run the same pipeline rather than a
    shortcut through it.
    """

    def run(table: PersonWindowTable) -> tuple[InfluenceEstimate, InfluenceEstimate]:
        folds = patient_folds_for(table, config)
        exposure = fit_balance_first_exposure(table, folds)
        trimmed = clip_propensity(exposure.propensity, config.clip_gamma)
        gate = build_certified_overlap_gate(
            table, trimmed.propensity, config.gate_grid, config.gate_tolerance
        )
        estimator = TargetEstimator(table=table, trimmed=trimmed, folds=folds, config=config)
        return (
            estimator.contrast(config.primary_outcome, gate.mask),
            estimator.contrast(config.secondary_outcome, gate.mask),
        )

    return run


def patient_folds_for(table: PersonWindowTable, config: AnalysisConfig) -> IntArray:
    """Fold assignment for the table in hand.

    The configured fold count is an upper bound: a configuration that holds a site out has
    fewer patients to split, and asking for more folds than there are patients would leave
    a fold empty rather than fail loudly.
    """
    patients = int(np.unique(table.patient_id).shape[0])
    if patients < 2:
        raise ValueError("the analysis needs at least two patients to cross-fit on")
    folds, _ = site_disjoint_patient_folds(
        table, max(2, min(config.n_folds, patients)), config.seed
    )
    return folds


def outcome_of(table: PersonWindowTable, name: str) -> FloatArray:
    """Resolve an outcome name to a vector, whether it is an outcome or a covariate."""
    if name in table.negative_controls or name in table.safety:
        return table.outcome(name)
    if name in {"takeover", "completion", "hand_back"}:
        return table.outcome(name)
    if name in table.covariates:
        return table.covariates[name]
    raise KeyError(f"unknown target for the estimator: {name}")


@dataclass
class TargetEstimator:
    """The primary one-step estimator applied to any named target on any region.

    The nuisance pair depends only on the target, not on the region, so it is fitted once
    per target and reused across the cohort and every stratum. That keeps the stratum rows
    and the headline on one estimator and one nuisance fit.
    """

    table: PersonWindowTable
    trimmed: TrimResult
    folds: IntArray
    config: AnalysisConfig
    _cache: dict[str, CrossFittedNuisance] = field(default_factory=dict)

    def nuisance(self, name: str) -> CrossFittedNuisance:
        if name not in self._cache:
            self._cache[name] = cross_fitted_nuisance(
                design=table_design(self.table, None),
                alert=self.table.alert.astype(np.int64),
                outcome=outcome_of(self.table, name),
                propensity=self.trimmed.propensity,
                folds=self.folds,
                learner=self.config.outcome_learner,
            )
        return self._cache[name]

    def contrast(self, name: str, mask: BoolArray) -> InfluenceEstimate:
        nuisance = self.nuisance(name)
        return aipw_contrast(
            f"aipw:{name}",
            self.table.alert.astype(np.float64),
            outcome_of(self.table, name),
            nuisance.propensity,
            nuisance.mu_treated,
            nuisance.mu_control,
            mask,
        )

    def contrasts(self, name: str, masks: dict[str, BoolArray]) -> dict[str, InfluenceEstimate]:
        return {label: self.contrast(name, mask) for label, mask in masks.items()}


def run_analysis(
    table: PersonWindowTable,
    config: AnalysisConfig,
    reader_arms: list[ArmCounts] | None = None,
) -> AnalysisResult:
    """Run the pre-specified pipeline on a person-window table."""
    _LOG.info("analysis start: %d windows over %d sites", table.size, table.n_sites)

    folds = patient_folds_for(table, config)
    exposure = fit_balance_first_exposure(table, folds)
    trimmed = clip_propensity(exposure.propensity, config.clip_gamma)
    gate = build_certified_overlap_gate(
        table, trimmed.propensity, config.gate_grid, config.gate_tolerance
    )
    _LOG.info(
        "certified gate at eps %.4f retains %.3f of the cohort", gate.epsilon, gate.retained_share
    )

    estimator = TargetEstimator(table=table, trimmed=trimmed, folds=folds, config=config)
    primary = estimator.contrast(config.primary_outcome, gate.mask)
    secondary = estimator.contrast(config.secondary_outcome, gate.mask)

    mediation = cross_fitted_mediation_nuisance(
        table, trimmed.propensity, folds, config.secondary_outcome, config.outcome_learner
    )
    paths, mediator_diagnostics = cross_fitted_decomposition(
        table, mediation, gate.mask, config.secondary_outcome
    )
    theta = mediated_fraction(paths)

    falsification = run_falsification(
        primary,
        _unexposed_risk(table, config.primary_outcome, gate.mask),
        lambda name: estimator.contrast(name, gate.mask),
    )

    ablation = run_ablation_grid(
        table,
        folds,
        gate.mask,
        (config.primary_outcome, config.secondary_outcome),
        config.clip_gamma,
    )
    ablation_summary = _ablation_summary(ablation, gate)

    strata, pooled, gradient = _strata(table, config, estimator, gate, mediation)
    gap = maximum_subgroup_gap(
        [row.label for row in strata],
        np.asarray([row.takeover for row in strata], dtype=np.float64),
    )

    pipeline = contrast_pair(config)
    benefit_curve = causal_benefit_curve(table, table.alert_score, pipeline, config.thresholds)
    operating_point = select_operating_point(benefit_curve)
    band_comparison = abstention_band_comparison(
        table, table.alert_score, table.alert, config.abstention, operating_point.threshold
    )
    attenuation = window_attenuation(
        table,
        folds,
        config.secondary_outcome,
        config.outcome_learner,
        gate.mask,
        config.clip_gamma,
        config.window_multipliers,
        propensity=trimmed.propensity,
    )
    alternatives = _alternatives(table, config, folds, gate.mask, primary, paths, pipeline)

    success = success_criteria(
        proportion_of_flag(table.safety["technical_success"].astype(np.bool_)),
        secondary,
        gap.gap,
    )
    reader = {"available": False, "reason": "no reader layer supplied to this run"}
    if reader_arms is not None:
        reader = {"available": True, **reader_comparison(reader_arms)}
    table_record = main_comparison_table(
        table,
        exposure.inverse_probability_weights(table.alert),
        primary,
        secondary,
        paths.as_dict(),
        {"headroom_note": "the ceiling needs the reference channel; see evaluation.headroom"},
        reader,
    )
    family_multiplicity = _multiplicity(primary, secondary, strata, config)

    result = AnalysisResult(
        config=config,
        exposure=exposure,
        gate=gate,
        trimmed=trimmed,
        probes=_probes(table, trimmed, gate, falsification, config),
        primary=primary,
        secondary=secondary,
        paths=paths,
        theta=theta,
        mediator_diagnostics=mediator_diagnostics,
        theta_test=theta_null_test(theta),
        falsification=falsification,
        ablation=ablation,
        ablation_summary=ablation_summary,
        strata=strata,
        pooled=pooled,
        gradient=gradient,
        subgroup_gap=gap.as_dict(),
        alternatives=alternatives,
        attenuation=attenuation,
        benefit_curve=benefit_curve,
        operating_point=operating_point,
        band_comparison=band_comparison,
        success=success,
        table_record=table_record,
        family_multiplicity=family_multiplicity,
        reader=reader,
    )
    _LOG.info(
        "analysis done: takeover %+.2f pp, completion %+.2f pp, theta %.3f",
        100.0 * primary.point,
        100.0 * secondary.point,
        theta.point,
    )
    return result


def _probes(
    table: PersonWindowTable,
    trimmed: TrimResult,
    gate: CertifiedGate,
    falsification: FalsificationReport,
    config: AnalysisConfig,
) -> list[ProbeResult]:
    """One probe per identification assumption, in the order Sec. 4.3 lists them."""
    p_values = np.asarray([row.estimate.p_value() for row in falsification.panel], dtype=np.float64)
    return [
        consistency_probe(table, config.consistency_floor),
        positivity_probe(trimmed.propensity, gate.epsilon, config.gate_ess_floor),
        exchangeability_probe(p_values, config.alpha),
    ]


def _unexposed_risk(table: PersonWindowTable, outcome_name: str, mask: BoolArray) -> float:
    selected = mask & ~table.alert
    if not bool(selected.any()):
        raise ValueError("the certified region has no unexposed windows")
    return float(np.mean(outcome_of(table, outcome_name)[selected]))


def _ablation_summary(rows: list[AblationRow], gate: CertifiedGate) -> dict[str, float]:
    return {
        "takeover_deficit_price_pp": ablation_deficit_price(rows, "takeover"),
        "completion_deficit_price_pp": ablation_deficit_price(rows, "completion"),
        "estimator_spread_takeover_pp": estimator_spread(rows, "takeover"),
        "estimator_spread_completion_pp": estimator_spread(rows, "completion"),
        "nuisance_drift_takeover_pp": nuisance_drift(rows, "takeover"),
        "nuisance_drift_completion_pp": nuisance_drift(rows, "completion"),
        "gate_epsilon": gate.epsilon,
        "gate_deficit_flagged": float(gate.deficit_flagged),
    }


def _strata(
    table: PersonWindowTable,
    config: AnalysisConfig,
    estimator: TargetEstimator,
    gate: CertifiedGate,
    mediation: MediationNuisance,
) -> tuple[list[StratumRow], list[dict[str, object]], GradientTest]:
    """The per-condition breakdown and the forest's mediated fractions.

    The causal contrasts of a stratum are the one-step estimator evaluated on the
    intersection of the certified region with the stratum, so a row and the headline share
    one estimator and one nuisance fit. The mediated fraction comes from the honest-split
    causal forest of Algorithm 4 instead, because that is the estimator the manuscript uses
    for the per-stratum decomposition.
    """
    terms_total = _forest_terms(
        table, mediation, estimator.trimmed.propensity, config, kind="total"
    )
    terms_mediated = _forest_terms(
        table, mediation, estimator.trimmed.propensity, config, kind="mediated"
    )
    design = table.full_design()

    groups: list[tuple[str, str, BoolArray]] = []
    for stratum in complexity_partition(table):
        groups.append(("complexity", stratum.label, stratum.mask))
    for stratum in vessel_bed_partition(table):
        if stratum.label == "Infrapopliteal":
            groups.append(("vessel_bed", stratum.label, stratum.mask))
    for stratum in experience_partition(table):
        groups.append(("operator_experience", stratum.label, stratum.mask))
    for stratum in site_partition(table):
        groups.append(("site", stratum.label, stratum.mask))

    forest_masks = [(label, mask) for _, label, mask in groups]
    forest = HonestSplitCausalForest(config.forest).fit(
        design,
        table.alert.astype(np.float64),
        outcome_of(table, config.secondary_outcome),
        gate.mask,
        table.patient_id,
    )
    fractions = forest_strata(forest, design, terms_total, terms_mediated, forest_masks)

    rows: list[StratumRow] = []
    for group, label, mask in groups:
        region = gate.mask & mask
        if int(np.count_nonzero(region)) < 2:
            continue
        takeover = estimator.contrast(config.primary_outcome, region)
        completion = estimator.contrast(config.secondary_outcome, region)
        fraction = fractions.get(label)
        rows.append(
            StratumRow(
                label=label,
                group=group,
                n=int(np.count_nonzero(mask)),
                takeover=100.0 * takeover.point,
                takeover_se=100.0 * takeover.standard_error,
                takeover_p=takeover.p_value(),
                completion=100.0 * completion.point,
                completion_se=100.0 * completion.standard_error,
                completion_p=completion.p_value(),
                mediated_fraction=float("nan") if fraction is None else fraction.point,
                mediated_fraction_se=float("nan") if fraction is None else fraction.standard_error,
                silent_override=silent_override_rate(table, mask),
            )
        )

    ordered_labels = [
        stratum.label for stratum in complexity_partition(table) if stratum.label in fractions
    ]
    gradient = monotone_gradient_test(
        ordered_labels,
        [fractions[label].point for label in ordered_labels],
        [fractions[label].standard_error for label in ordered_labels],
    )
    pooled_rows = _pooled_rows(rows)
    return rows, pooled_rows, gradient


def _pooled_rows(rows: list[StratumRow]) -> list[dict[str, object]]:
    """Size-weighted pooling of the two pooled complexity rows of Table 3.

    Ref: Table 3, pp. 9-10. The printed sizes of the two pooled rows add to the analytical
    cohort, so the infrapopliteal stratum is read as disjoint from the three complexity
    classes and the two rows partition the strata they pool.
    """
    labels = [row.label for row in rows]
    position = {label: index for index, label in enumerate(labels)}
    groups = [
        (
            "Classes I-II pooled",
            [position[label] for label in ("Class I", "Class II") if label in position],
        ),
        (
            "Class III and infrapopliteal pooled",
            [position[label] for label in ("Class III", "Infrapopliteal") if label in position],
        ),
    ]
    usable = [(name, members) for name, members in groups if members]
    if not usable:
        return []
    return pooled_stratum_rows(
        usable,
        np.asarray([row.takeover for row in rows], dtype=np.float64),
        np.asarray([row.completion for row in rows], dtype=np.float64),
        np.asarray([row.mediated_fraction for row in rows], dtype=np.float64),
        np.asarray([float(row.n) for row in rows], dtype=np.float64),
    )


def _forest_terms(
    table: PersonWindowTable,
    nuisance: MediationNuisance,
    propensity: FloatArray,
    config: AnalysisConfig,
    kind: str,
) -> FloatArray:
    """Row terms of the interventional decomposition, used as forest inputs."""
    alert = table.alert.astype(np.float64)
    mediator = table.hand_back.astype(np.float64)
    outcome = outcome_of(table, config.secondary_outcome)
    observed = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_exposed,
        nuisance.mediator_exposed,
    )
    if kind == "total":
        comparison = augmented_psi_terms(
            0,
            alert,
            outcome,
            mediator,
            propensity,
            nuisance.cell(0, 0),
            nuisance.cell(0, 1),
            nuisance.mediator_control,
            nuisance.mediator_control,
        )
    elif kind == "mediated":
        comparison = augmented_psi_terms(
            1,
            alert,
            outcome,
            mediator,
            propensity,
            nuisance.cell(1, 0),
            nuisance.cell(1, 1),
            nuisance.mediator_control,
            nuisance.mediator_exposed,
        )
    else:
        raise KeyError(f"unknown forest term family: {kind}")
    return np.asarray(observed - comparison, dtype=np.float64)


def _alternatives(
    table: PersonWindowTable,
    config: AnalysisConfig,
    folds: IntArray,
    mask: BoolArray,
    primary: InfluenceEstimate,
    paths: PathEstimates,
    pipeline: ContrastPair,
) -> dict[str, object]:
    """The Table 4 Panel E rows."""
    design = table.full_design()
    alert = table.alert.astype(np.int64)
    outcome = outcome_of(table, config.primary_outcome)
    mu_treated, mu_control = cross_fitted_outcome(
        design, alert, outcome, folds, config.outcome_learner
    )
    shift = solve_shift_for_marginal_rate(table.alert_score, table.alert.astype(np.float64))
    stochastic = stochastic_intervention_contrast(
        "stochastic_intervention",
        table.alert.astype(np.float64),
        outcome,
        table.alert_score,
        mu_treated,
        mu_control,
        mask,
        shift,
    )
    return {
        "shift": shift,
        "stochastic_intervention": stochastic.as_dict(),
        "cluster_robust": cluster_robust_rows(primary, table, mask),
        "bootstrap": bootstrap_row(primary, config.n_bootstrap, config.bootstrap_seed),
        "leave_one_site_out": leave_one_site_out(table, pipeline, 0),
        "additivity_ok": float(
            abs(paths.total.point - (paths.direct.point + paths.mediated.point)) < 1e-9
        ),
    }


def _multiplicity(
    primary: InfluenceEstimate,
    secondary: InfluenceEstimate,
    strata: list[StratumRow],
    config: AnalysisConfig,
) -> dict[str, object]:
    """Holm step-down on the primary family, as the pre-specified correction requires."""
    family = np.asarray([primary.p_value(), secondary.p_value()], dtype=np.float64)
    corrected = holm_step_down(family, config.alpha)
    return {
        "family": corrected.family,
        "raw": [float(value) for value in corrected.raw],
        "adjusted": [float(value) for value in corrected.adjusted],
        "rejected": [bool(value) for value in corrected.rejected],
        "subgroup_family_size": float(len(strata)),
        "negative_control_family_size": float(len(NEGATIVE_CONTROLS)),
    }


def describe_workflow(table: PersonWindowTable, mask: BoolArray) -> dict[str, object]:
    """The workflow block of Sec. 2.7, p. 5."""
    return {
        "fluoroscopy_time_min": describe(table, "fluoroscopy_time_min", mask).as_dict(),
        "contrast_volume_ml": describe(table, "contrast_volume_ml", mask).as_dict(),
        "procedure_duration_min": describe(table, "procedure_duration_min", mask).as_dict(),
        "integration_cost": integration_cost(table),
    }


def panel_sizes(table: PersonWindowTable) -> dict[str, list[dict[str, object]]]:
    """Sizes of every stratum the printed panels tabulate."""
    return {
        "complexity": [
            {"label": stratum.label, "n": stratum.n} for stratum in complexity_partition(table)
        ],
        "pooled_complexity": [
            {"label": stratum.label, "n": stratum.n}
            for stratum in pooled_complexity_partition(table)
        ],
        "experience": [
            {"label": stratum.label, "n": stratum.n} for stratum in experience_partition(table)
        ],
        "site": [{"label": stratum.label, "n": stratum.n} for stratum in site_partition(table)],
        "volume_tertile": [
            {"label": stratum.label, "n": stratum.n} for stratum in volume_tertile_partition(table)
        ],
    }


__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ContrastPair",
    "StratumRow",
    "contrast_pair",
    "describe_workflow",
    "outcome_of",
    "panel_sizes",
    "patient_folds_for",
    "run_analysis",
]
