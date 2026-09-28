"""Second verification layer: execution.

Ref: none in the manuscript - this layer establishes that the release runs. It reads a
staged cohort root when one is supplied, runs a forward and a backward pass, updates
parameters, round-trips a checkpoint, and checks the algebraic identities the manuscript
states against the code that produces them.

Each check is written so that its verification logic is independent of the code it
verifies: the identities are re-derived here from the definitions rather than called back
out of the modules being tested. A check that cannot be executed on the material at hand is
recorded as BLOCKED or NOT_RUN and never as a pass.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..analysis.pipeline import AnalysisConfig, run_analysis
from ..cohort.builder import build_person_windows
from ..cohort.layout import CohortLayout
from ..estimands.attenuation import monotone_attenuation
from ..estimands.interventional import augmented_psi_terms, psi_estimate
from ..estimators.aipw import eq2_terms
from ..estimators.crossfit import clip_propensity
from ..estimators.decomposition import cross_fitted_decomposition, cross_fitted_mediation_nuisance
from ..estimators.influence import ratio
from ..exposure.model import fit_balance_first_exposure
from ..forest.causal_forest import ForestConfig
from ..forest.gradient import monotone_gradient_test
from ..multiplicity.corrections import benjamini_hochberg, holm_step_down
from ..overlap.effective_sample_size import kish_effective_sample_size
from ..overlap.gate import (
    build_certified_overlap_gate,
    gate_mask,
    positivity_deficit_price,
)
from ..perception.checkpoint import load_checkpoint, save_checkpoint
from ..perception.data import FluoroscopyWindowDataset, FrameStore, site_order
from ..perception.dice import dice_coefficient
from ..perception.losses import policy_loss
from ..perception.spatial_predicate import AlertPolicyNet
from ..perception.training import AlertTrainingConfig, score_windows, train_alert_policy
from ..protocol.folds import site_disjoint_patient_folds
from ..protocol.schema import COVARIATE_BLOCKS
from ..protocol.windowing import WindowSpec
from ..reporting.success import wilson_interval
from ..strata.heterogeneity import cochran_q
from ..utils.logging import get_logger
from .conformance import FAIL, NOT_RUN, PASS

_LOG = get_logger("verification.execution")

BLOCKED = "BLOCKED"

# Comparison tolerances. They are set from the sampling error of the quantity being
# compared, not from the size of the number, so a narrowly estimated quantity is held to a
# tighter standard than a noisy one.
IDENTITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ExecutionCheck:
    """One executed check and what it observed."""

    name: str
    status: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "values": self.values,
        }


@dataclass
class ExecutionReport:
    """The whole second layer."""

    checks: list[ExecutionCheck] = field(default_factory=list)
    cohort_root: str = ""
    cohort_label: str = "no cohort staged"

    def add(self, check: ExecutionCheck) -> None:
        self.checks.append(check)

    @property
    def summary(self) -> dict[str, float]:
        counts = {PASS: 0, FAIL: 0, NOT_RUN: 0, BLOCKED: 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return {
            "checks": float(len(self.checks)),
            "pass": float(counts[PASS]),
            "fail": float(counts[FAIL]),
            "not_run": float(counts[NOT_RUN]),
            "blocked": float(counts[BLOCKED]),
        }

    @property
    def code_status(self) -> str:
        return FAIL if self.summary["fail"] > 0 else PASS

    @property
    def overall_status(self) -> str:
        if self.summary["fail"] > 0:
            return "FAIL"
        if self.summary["blocked"] > 0 or self.summary["not_run"] > 0:
            return "PARTIALLY_VERIFIED"
        return "VERIFIED"

    def as_dict(self) -> dict[str, object]:
        return {
            "cohort_root_supplied": bool(self.cohort_root),
            "cohort_label": self.cohort_label,
            "summary": self.summary,
            "code_status": self.code_status,
            "overall_status": self.overall_status,
            "checks": [check.as_dict() for check in self.checks],
        }


def _check_reading(root: Path) -> ExecutionCheck:
    """Read the cohort contract and assert the schema of the built table."""
    layout = CohortLayout(root=root)
    missing = layout.missing()
    if missing:
        return ExecutionCheck(
            "read_cohort_layers", BLOCKED, "cohort root is not staged", {"missing": missing}
        )
    spec = WindowSpec(length_s=30.0, stride_s=30.0, response_s=30.0)
    table = build_person_windows(layout, spec)
    members = [name for block in COVARIATE_BLOCKS.values() for name in block]
    non_finite = [name for name in members if not bool(np.isfinite(table.covariates[name]).all())]
    sizes = {
        "rows": table.size,
        "patients": table.n_patients,
        "sites": table.n_sites,
        "alert_rate": float(np.mean(table.alert)),
    }
    if non_finite or table.size == 0:
        return ExecutionCheck(
            "read_cohort_layers", FAIL, f"non-finite covariates: {non_finite}", sizes
        )
    return ExecutionCheck("read_cohort_layers", PASS, "both layers read and the table built", sizes)


def _check_forward_and_backward() -> list[ExecutionCheck]:
    """Forward, loss, backward and one parameter update on the alert policy."""
    checks: list[ExecutionCheck] = []
    torch.set_num_threads(1)
    model = AlertPolicyNet()
    frames = torch.randn(2, 4, 1, 32, 32)
    output = model(frames)
    shapes = {
        "predicate": list(output.predicate_logits.shape),
        "alert": list(output.alert_logits.shape),
        "window_score": list(output.window_scores().shape),
    }
    expected = [2, 4, 1, 32, 32]
    checks.append(
        ExecutionCheck(
            "tensor_forward_shapes",
            PASS if list(output.predicate_logits.shape) == expected else FAIL,
            "predicate logits must keep the frame grid",
            shapes,
        )
    )

    mask = (torch.rand(2, 4, 1, 32, 32) > 0.7).float()
    frame_alert = (torch.rand(2, 4) > 0.5).float()
    window_alert = (torch.rand(2, 1) > 0.5).float()
    # Every frame of the constructed batch carries a reference mask, so the Dice term sees
    # the whole batch rather than a spatially sampled subset of it.
    breakdown = policy_loss(
        output,
        mask,
        frame_alert,
        window_alert,
        1.0,
        1.0,
        0.5,
        1.0,
        mask_observed=torch.ones(2, 4, dtype=torch.bool),
    )
    before = float(breakdown.total.item())
    breakdown.total.backward()  # type: ignore[no-untyped-call]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    grad_norm = float(sum(float(g.abs().sum()) for g in gradients))
    checks.append(
        ExecutionCheck(
            "loss_and_backward",
            PASS if math.isfinite(before) and grad_norm > 0.0 else FAIL,
            "loss finite and gradients propagated to every parameter",
            {"loss": before, "gradient_abs_sum": grad_norm},
        )
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    snapshot = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    moved = sum(
        1
        for stale, parameter in zip(snapshot, model.parameters(), strict=False)
        if not torch.equal(stale, parameter.detach())
    )
    checks.append(
        ExecutionCheck(
            "parameter_update",
            PASS if moved == len(snapshot) else FAIL,
            "every parameter moved on the optimizer step",
            {"trainable_tensors": len(snapshot), "moved": moved},
        )
    )
    return checks


def _check_checkpoint() -> ExecutionCheck:
    """Checkpoint round trip reproduces the forward pass exactly."""
    torch.set_num_threads(1)
    model = AlertPolicyNet()
    frames = torch.randn(1, 2, 1, 16, 16)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pt"
        save_checkpoint(path, model, None, 3, 1234, extra={"note": "round trip"})
        restored = AlertPolicyNet()
        epoch, seed, extra = load_checkpoint(path, restored, None)
    model.eval()
    restored.eval()
    with torch.no_grad():
        original = model(frames).window_scores()
        copied = restored(frames).window_scores()
    identical = bool(torch.equal(original, copied))
    return ExecutionCheck(
        "checkpoint_round_trip",
        PASS if identical and epoch == 3 and seed == 1234 else FAIL,
        "saved weights reproduce the scored stream and carry the seed",
        {"epoch": epoch, "seed": seed, "extra": extra, "identical": identical},
    )


def _check_overfit() -> ExecutionCheck:
    """One batch overfits: the loss must fall sharply on a single repeated batch."""
    torch.set_num_threads(1)
    frames = torch.randn(4, 4, 1, 32, 32)
    mask = torch.zeros(4, 4, 1, 32, 32)
    mask[:, :, :, 8:24, 8:24] = 1.0
    labels = torch.zeros(4, 4)
    labels[:, 0] = 1.0
    window = torch.ones(4, 1)
    model = AlertPolicyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = 0.0
    last = 0.0
    for step in range(40):
        output = model(frames)
        breakdown = policy_loss(
            output,
            mask,
            labels,
            window,
            1.0,
            1.0,
            0.5,
            1.0,
            mask_observed=torch.ones(4, 4, dtype=torch.bool),
        )
        optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()  # type: ignore[no-untyped-call]  # type: ignore[no-untyped-call]
        optimizer.step()
        value = float(breakdown.total.item())
        if step == 0:
            first = value
        last = value
    ratio_value = last / first if first > 0.0 else float("inf")
    return ExecutionCheck(
        "single_batch_overfit",
        PASS if last < 0.5 * first else FAIL,
        "loss on a repeated batch must fall well below its first-epoch value",
        {"first": first, "last": last, "ratio": ratio_value},
    )


def _check_training_loop(root: Path) -> ExecutionCheck:
    """A short training loop over the staged sample, with the checkpoint it writes."""
    torch.set_num_threads(1)
    table = _staged_table(root)
    if table is None:
        return ExecutionCheck("training_loop", BLOCKED, "no cohort root was supplied", {})
    store = FrameStore(root=root, image_size=(32, 32))
    dataset = FluoroscopyWindowDataset(
        table=table, store=store, window_frames=4, sites=site_order(table)
    )
    model = AlertPolicyNet()
    config = AlertTrainingConfig(
        epochs=2, batch_size=8, learning_rate=1e-3, warmup_epochs=1, checkpoint_every=10
    )
    trace = train_alert_policy(
        model, dataset, config, device="cpu", checkpoint_path=root / "policy.pt"
    )
    scores, states, rows, dice = score_windows(model, dataset, batch_size=16)
    agreement = float(np.mean((scores >= 0.4) == states))
    passed = trace.decreased and scores.shape[0] == rows.shape[0]
    return ExecutionCheck(
        "training_loop",
        PASS if passed else FAIL,
        "two epochs on the staged sample, then score the same windows",
        {
            "steps": trace.steps,
            "loss_first": trace.total[0],
            "loss_last": trace.total[-1],
            "decreased": trace.decreased,
            "scored_windows": int(scores.shape[0]),
            "predicate_dice": float(dice.mean()) if dice.size else float("nan"),
            "policy_edition_agreement": agreement,
        },
    )


def _staged_table(root: Path) -> Any:
    """Build the person-window table of whatever root is staged, or ``None`` if none is."""
    layout = CohortLayout(root=root)
    if layout.missing():
        return None
    return build_person_windows(layout, WindowSpec(length_s=30.0, stride_s=30.0, response_s=30.0))


def _check_decomposition(root: Path) -> list[ExecutionCheck]:
    """Additivity, the reduction to AIPW, and recovery of the mediated fraction."""
    checks: list[ExecutionCheck] = []
    table = _staged_table(root)
    if table is None:
        return [ExecutionCheck("decomposition", BLOCKED, "no cohort root was supplied", {})]
    folds, _ = site_disjoint_patient_folds(table, _fold_count(table), 20260101)
    exposure = fit_balance_first_exposure(table, folds)
    trimmed = clip_propensity(exposure.propensity, 0.01)
    gate = build_certified_overlap_gate(table, trimmed.propensity)
    nuisance = cross_fitted_mediation_nuisance(
        table, trimmed.propensity, folds, "completion", "glm"
    )
    paths, _ = cross_fitted_decomposition(table, nuisance, gate.mask, "completion")
    residual = paths.total.point - (paths.direct.point + paths.mediated.point)
    checks.append(
        ExecutionCheck(
            "decomposition_additivity",
            PASS if abs(residual) <= IDENTITY_TOLERANCE else FAIL,
            "total must equal direct plus mediated exactly, since the two are one difference",
            {"residual": residual},
        )
    )

    alert = table.alert.astype(np.float64)
    outcome = table.outcome("completion")
    mediator = table.hand_back.astype(np.float64)
    terms = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        nuisance.propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_exposed,
        nuisance.mediator_exposed,
    )
    hand_written = np.where(
        gate.mask,
        alert
        * (outcome - np.where(mediator > 0.5, nuisance.cell(1, 1), nuisance.cell(1, 0)))
        / nuisance.propensity
        + nuisance.mediator_exposed * nuisance.cell(1, 1)
        + (1.0 - nuisance.mediator_exposed) * nuisance.cell(1, 0),
        0.0,
    )
    arm_mean = float(hand_written.sum() / np.count_nonzero(gate.mask))
    gap = abs(psi_estimate(terms, gate.mask).point - arm_mean)
    checks.append(
        ExecutionCheck(
            "decomposition_reduces_to_arm_mean",
            PASS if gap <= 1e-9 else FAIL,
            "with the target rule equal to the observed one the term must collapse to the AIPW arm mean",
            {"gap": gap, "arm_mean": arm_mean},
        )
    )

    theta = ratio(paths.mediated, paths.total, name="mediated_fraction")
    hand_ratio = (
        paths.mediated.point / paths.total.point if abs(paths.total.point) > 0 else float("nan")
    )
    checks.append(
        ExecutionCheck(
            "mediated_fraction_ratio",
            PASS if abs(theta.point - hand_ratio) <= 1e-12 else FAIL,
            "the fraction must equal the mediated component over the total",
            {"fraction": theta.point, "hand_ratio": hand_ratio},
        )
    )
    return checks


def _check_eq2_identity() -> ExecutionCheck:
    """Eq. (2) with a flat outcome regression must equal the hand-written IPW arms."""
    rng = np.random.default_rng(19)
    size = 400
    alert = rng.binomial(1, 0.4, size=size).astype(np.float64)
    outcome = rng.binomial(1, 0.3, size=size).astype(np.float64)
    propensity = np.clip(rng.uniform(0.15, 0.85, size=size), 0.05, 0.95)
    flat = np.full(size, float(outcome.mean()))
    terms = (
        alert * (outcome - flat) / propensity
        - (1.0 - alert) * (outcome - flat) / (1.0 - propensity)
        + flat
        - flat
    )
    produced = eq2_terms(alert, outcome, propensity, flat, flat)
    gap = float(np.max(np.abs(produced - terms)))
    return ExecutionCheck(
        "eq2_matches_hand_written_terms",
        PASS if gap <= 1e-12 else FAIL,
        "the bracketed term of Eq. (2) against an independent transcription",
        {"max_abs_gap": gap},
    )


def _check_estimator_utilities() -> list[ExecutionCheck]:
    """Utilities whose closed forms can be written down by hand."""
    checks: list[ExecutionCheck] = []

    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    hand_ess = weights.sum() ** 2 / (weights**2).sum()
    produced = kish_effective_sample_size(weights)
    checks.append(
        ExecutionCheck(
            "kish_effective_sample_size",
            PASS if abs(produced - hand_ess) <= 1e-12 else FAIL,
            "closed form (sum w)^2 / sum w^2",
            {"produced": produced, "hand": hand_ess},
        )
    )

    propensity = np.asarray([0.01, 0.2, 0.5, 0.79, 0.99])
    produced_mask = gate_mask(propensity, 0.2)
    hand_mask = np.asarray([0.2 <= value <= 0.8 for value in propensity])
    checks.append(
        ExecutionCheck(
            "gate_mask_region",
            PASS if bool(np.array_equal(produced_mask, hand_mask)) else FAIL,
            "the gate keeps exactly the rows with eps <= pi <= 1 - eps",
            {"kept": int(np.count_nonzero(produced_mask))},
        )
    )
    price = positivity_deficit_price(4.1, 3.2)
    checks.append(
        ExecutionCheck(
            "positivity_deficit_price",
            PASS if abs(price - 0.9) <= 1e-12 else FAIL,
            "the price is the gated contrast minus the ungated contrast",
            {"price": price},
        )
    )

    p_values = np.asarray([0.001, 0.02, 0.2, 0.04])
    holm = holm_step_down(p_values, 0.05)
    hand_holm = np.asarray([0.004, 0.06, 0.2, 0.08])
    checks.append(
        ExecutionCheck(
            "holm_step_down",
            PASS if float(np.max(np.abs(holm.adjusted - hand_holm))) <= 1e-12 else FAIL,
            "step-down adjustment against hand-computed values",
            {"produced": [float(v) for v in holm.adjusted], "hand": [float(v) for v in hand_holm]},
        )
    )
    bh = benjamini_hochberg(p_values, 0.05)
    hand_bh = np.asarray([0.004, 0.04, 0.2, 0.05333333333333334])
    checks.append(
        ExecutionCheck(
            "benjamini_hochberg",
            PASS if float(np.max(np.abs(bh.adjusted - hand_bh))) <= 1e-9 else FAIL,
            "step-up adjustment against hand-computed values",
            {"produced": [float(v) for v in bh.adjusted], "hand": [float(v) for v in hand_bh]},
        )
    )

    low, high = wilson_interval(30, 100)
    z_score = 1.959963984540054
    share = 0.3
    denominator = 1.0 + z_score**2 / 100.0
    hand_centre = (share + z_score**2 / 200.0) / denominator
    hand_half = (
        z_score
        * math.sqrt(share * (1.0 - share) / 100.0 + z_score**2 / (4.0 * 100.0 * 100.0))
        / denominator
    )
    gap = max(abs(low - (hand_centre - hand_half)), abs(high - (hand_centre + hand_half)))
    checks.append(
        ExecutionCheck(
            "wilson_interval",
            PASS if gap <= 1e-6 else FAIL,
            "score interval against the closed form at 95%",
            {"low": low, "high": high, "max_gap": gap},
        )
    )

    q, df, p_value, pooled = cochran_q(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.0, 1.0]))
    checks.append(
        ExecutionCheck(
            "cochran_q",
            PASS if abs(q - 2.0) <= 1e-12 and abs(pooled - 2.0) <= 1e-12 and df == 2 else FAIL,
            "equal-weight case has a closed form",
            {"q": q, "df": df, "p_value": p_value, "pooled": pooled},
        )
    )

    prediction = torch.tensor([[[[10.0, -10.0]]]])
    reference = torch.tensor([[[[1.0, 0.0]]]])
    dice = dice_coefficient(prediction, reference)
    checks.append(
        ExecutionCheck(
            "dice_coefficient",
            PASS if abs(dice - 1.0) <= 1e-9 else FAIL,
            "a perfect mask scores one",
            {"dice": dice},
        )
    )

    gradient = monotone_gradient_test(["a", "b", "c"], [0.2, 0.4, 0.8], [0.05, 0.05, 0.05])
    order = np.asarray([0.0, 1.0, 2.0])
    centred = order - order.mean()
    hand_slope = float(np.sum(centred * np.asarray([0.2, 0.4, 0.8])) / np.sum(centred**2))
    checks.append(
        ExecutionCheck(
            "gradient_slope",
            PASS if abs(gradient.slope - hand_slope) <= 1e-12 else FAIL,
            "least-squares slope on the order index",
            {"slope": gradient.slope, "hand": hand_slope},
        )
    )
    return checks


def _fold_count(table: Any) -> int:
    """The largest fold count the staged table can carry."""
    return max(2, min(5, int(np.unique(table.patient_id).shape[0])))


def _pipeline_config(table: Any) -> AnalysisConfig:
    """A configuration the staged table can actually carry.

    The fold count cannot exceed the number of patients in the staged sample, and the forest
    needs two rows per leaf, so both are read off the table rather than fixed here.
    """
    folds = _fold_count(table)
    rows_per_half = max(2, table.size // 4)
    return AnalysisConfig(
        n_folds=folds,
        n_bootstrap=200,
        thresholds=(0.25, 0.35, 0.45),
        forest=ForestConfig(trees=24, min_leaf=rows_per_half, max_depth=3),
    )


def _check_pipeline_identities(root: Path) -> list[ExecutionCheck]:
    """Run the whole pipeline on the staged sample and check its internal consistency."""
    table = _staged_table(root)
    if table is None:
        return [ExecutionCheck("pipeline_end_to_end", BLOCKED, "no cohort root was supplied", {})]
    config = _pipeline_config(table)
    result = run_analysis(table, config)
    checks: list[ExecutionCheck] = []
    residual = result.paths.total.point - (result.paths.direct.point + result.paths.mediated.point)
    checks.append(
        ExecutionCheck(
            "pipeline_decomposition_additivity",
            PASS if abs(residual) <= IDENTITY_TOLERANCE else FAIL,
            "the pipeline's own decomposition must be additive",
            {"residual": residual},
        )
    )
    fraction_gap = abs(result.theta.point - result.paths.mediated.point / result.paths.total.point)
    checks.append(
        ExecutionCheck(
            "pipeline_mediated_fraction",
            PASS if fraction_gap <= 1e-9 else FAIL,
            "the reported fraction must equal the ratio of the reported components",
            {"gap": fraction_gap},
        )
    )
    gated = result.ablation_summary["takeover_deficit_price_pp"]
    expected_price = 100.0 * (
        result.ablation[0].takeover.point
        - next(
            row for row in result.ablation if row.spec.label == "aipw_ungated_full_cohort"
        ).takeover.point
    )
    checks.append(
        ExecutionCheck(
            "pipeline_positivity_deficit_price",
            PASS if abs(gated - expected_price) <= 1e-9 else FAIL,
            "the recorded price must equal the gated minus ungated contrast",
            {"price_pp": gated, "hand_pp": expected_price},
        )
    )
    attenuation = monotone_attenuation(result.attenuation)
    checks.append(
        ExecutionCheck(
            "pipeline_attenuation_sweep",
            PASS if len(result.attenuation) == len(config.window_multipliers) else NOT_RUN,
            "the sweep must return one point per multiplier; the direction is reported, not asserted",
            {
                "multipliers": attenuation["multipliers"],
                "fractions": attenuation["fractions"],
                "monotone_decreasing": attenuation["monotone_decreasing"],
            },
        )
    )
    repeated = run_analysis(table, _pipeline_config(table))
    deterministic = (
        abs(repeated.primary.point - result.primary.point) <= 0.0
        and abs(repeated.theta.point - result.theta.point) <= 0.0
    )
    checks.append(
        ExecutionCheck(
            "pipeline_determinism",
            PASS if deterministic else FAIL,
            "two runs at the same seed must return identical headline numbers",
            {
                "first": result.primary.point,
                "second": repeated.primary.point,
                "fraction_first": result.theta.point,
                "fraction_second": repeated.theta.point,
            },
        )
    )
    return checks


def _check_cohort_claims() -> list[ExecutionCheck]:
    """Checks that need the private cohort, recorded as blocked rather than skipped."""
    return [
        ExecutionCheck(
            "cohort_alert_model_discrimination",
            BLOCKED,
            "AUROC, AUPRC, Brier and the Dice of the spatial predicate are measured on the "
            "held-out clinical stream, which is held under site data-sharing agreements",
            {
                "printed_auroc": 0.918,
                "printed_auprc": 0.674,
                "printed_brier": 0.132,
                "printed_dice": 0.809,
            },
        ),
        ExecutionCheck(
            "cohort_table1_contrasts",
            BLOCKED,
            "the printed contrasts cannot be recomputed without the cohort; the estimators "
            "that would produce them are exercised by the suite instead",
            {"printed_takeover_pp": 3.2, "printed_completion_pp": 2.6},
        ),
        ExecutionCheck(
            "cohort_certified_region",
            BLOCKED,
            "the certified-overlap region and its retained share are properties of the cohort",
            {"printed_identifiable_share": 0.701},
        ),
        ExecutionCheck(
            "deployment_latency_table_a1",
            BLOCKED,
            "the per-stage latencies are wall-clock on the deployment configuration, which is "
            "not the host that produced this report",
            {"printed_end_to_end_ms": 54.7, "printed_budget_share": 0.023},
        ),
    ]


def run_execution_verification(
    cohort_root: Path | None, cohort_label: str = "the cohort root named by the experiment config"
) -> ExecutionReport:
    """Run the whole second layer.

    ``cohort_label`` records *what* the root is, so a report produced against a staged
    sample cannot be read as a report produced against the cohort itself.
    """
    report = ExecutionReport(
        cohort_root="" if cohort_root is None else str(cohort_root),
        cohort_label=cohort_label if cohort_root is not None else "no cohort staged",
    )
    if cohort_root is None:
        report.add(ExecutionCheck("read_cohort_layers", BLOCKED, "no cohort root was supplied", {}))
    else:
        report.add(_check_reading(cohort_root))
    for check in _check_forward_and_backward():
        report.add(check)
    report.add(_check_checkpoint())
    report.add(_check_overfit())
    if cohort_root is not None:
        report.add(_check_training_loop(cohort_root))
    else:
        report.add(ExecutionCheck("training_loop", BLOCKED, "no cohort root was supplied", {}))
    if cohort_root is not None:
        for check in _check_decomposition(cohort_root):
            report.add(check)
        for check in _check_pipeline_identities(cohort_root):
            report.add(check)
    else:
        for name in (
            "decomposition_additivity",
            "decomposition_reduces_to_arm_mean",
            "mediated_fraction_ratio",
        ):
            report.add(ExecutionCheck(name, BLOCKED, "no cohort root was supplied", {}))
    report.add(_check_eq2_identity())
    for check in _check_estimator_utilities():
        report.add(check)
    for check in _check_cohort_claims():
        report.add(check)
    _LOG.info(
        "execution verification: %d checks, %d failed, %d blocked",
        report.summary["checks"],
        report.summary["fail"],
        report.summary["blocked"],
    )
    return report


__all__ = [
    "BLOCKED",
    "IDENTITY_TOLERANCE",
    "ExecutionCheck",
    "ExecutionReport",
    "run_execution_verification",
]
