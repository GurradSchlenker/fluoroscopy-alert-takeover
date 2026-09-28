"""Estimate the two causal contrasts of Eq. (2).

Ref: Eq. (2), p. 17; Sec. 4.4, pp. 16-18; Table 2, p. 8 for the estimator ablation.
"""

from __future__ import annotations

from ..analysis.pipeline import TargetEstimator, contrast_pair
from ..estimators.ablation import (
    ABLATION_GRID,
    ablation_deficit_price,
    estimator_spread,
    nuisance_drift,
    run_ablation_grid,
)
from ..estimators.crossfit import clip_propensity
from ..exposure.model import fit_balance_first_exposure
from ..inference.alternatives import bootstrap_row, cluster_robust_rows, leave_one_site_out
from ..overlap.gate import build_certified_overlap_gate, positivity_deficit_price
from ..protocol.folds import site_disjoint_patient_folds
from ..utils.config import dig
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Estimate both contrasts, then the ablation grid and the alternative-inference rows."""
    context = prepare_context("estimate", "Estimate the prespecified causal contrasts", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    config = analysis_config_from(context.config)
    folds, _ = site_disjoint_patient_folds(table, config.n_folds, config.seed)
    exposure = fit_balance_first_exposure(table, folds)
    trimmed = clip_propensity(exposure.propensity, config.clip_gamma)
    gate = build_certified_overlap_gate(
        table, trimmed.propensity, config.gate_grid, config.gate_tolerance
    )
    estimator = TargetEstimator(table=table, trimmed=trimmed, folds=folds, config=config)
    primary = estimator.contrast(config.primary_outcome, gate.mask)
    secondary = estimator.contrast(config.secondary_outcome, gate.mask)

    ablation = run_ablation_grid(
        table,
        folds,
        gate.mask,
        (config.primary_outcome, config.secondary_outcome),
        config.clip_gamma,
        grid=(
            ABLATION_GRID
            if bool(dig(context.config, "estimator.run_ablation", True))
            else ABLATION_GRID[:4]
        ),
    )
    gated = primary.point
    ungated = estimator.contrast(config.primary_outcome, table.alert | ~table.alert).point
    context.record(
        {
            "status": "PASS",
            "takeover": primary.as_dict(),
            "completion": secondary.as_dict(),
            "cluster_robust": cluster_robust_rows(primary, table, gate.mask),
            "bootstrap": bootstrap_row(primary, config.n_bootstrap, config.bootstrap_seed),
            "leave_one_site_out": leave_one_site_out(table, contrast_pair(config), 0),
            "positivity_deficit_price_pp": positivity_deficit_price(gated, ungated),
            "estimator_spread_takeover_pp": estimator_spread(ablation, "takeover"),
            "nuisance_drift_takeover_pp": nuisance_drift(ablation, "takeover"),
            "ablation_deficit_price_pp": ablation_deficit_price(ablation, "takeover"),
            "ablation": [row.as_dict() for row in ablation],
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
