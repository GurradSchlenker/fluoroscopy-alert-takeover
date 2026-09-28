"""Build the certified-overlap gate of Algorithm 1.

Ref: Algorithm 1, p. 16 - the family selection, the gate sweep, the effective-sample-size
trajectory, the covariate-marginal distance and the positivity-deficit price.
"""

from __future__ import annotations

from ..estimators.crossfit import clip_propensity
from ..exposure.model import exposure_side_balance, fit_balance_first_exposure
from ..overlap.gate import build_certified_overlap_gate
from ..protocol.folds import site_disjoint_patient_folds
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Select the exposure-model family, freeze the gate and record the sweep."""
    context = prepare_context("gate", "Build the certified-overlap gate", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    config = analysis_config_from(context.config)
    folds, fold_sizes = site_disjoint_patient_folds(table, config.n_folds, config.seed)
    exposure = fit_balance_first_exposure(table, folds)
    trimmed = clip_propensity(exposure.propensity, config.clip_gamma)
    gate = build_certified_overlap_gate(
        table, trimmed.propensity, config.gate_grid, config.gate_tolerance
    )
    context.record(
        {
            "status": "FAIL" if gate.deficit_flagged else "PASS",
            "exposure_model": exposure.as_dict(),
            "folds": fold_sizes,
            "trimmed_mass": trimmed.trimmed_mass,
            "balance": exposure_side_balance(exposure, table),
            "gate": gate.as_dict(),
            "note": (
                None
                if not gate.deficit_flagged
                else "no gate met the balance tolerance; the least unbalanced one was frozen"
            ),
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
