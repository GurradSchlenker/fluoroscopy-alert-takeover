"""Decompose the completion effect under the interventional analogue of Eq. (1).

Ref: Eq. (1), p. 16; Algorithm 3, p. 18; Table 4 Panel B, p. 11 for the window-length
attenuation prediction; Table 3, pp. 9-10 for the per-stratum fractions.
"""

from __future__ import annotations

from ..analysis.pipeline import run_analysis
from ..estimands.attenuation import monotone_attenuation
from ..estimands.interventional import theta_null_test
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Run the decomposition, the mediator diagnostics and the attenuation sweep."""
    context = prepare_context("decompose", "Decompose the completion effect", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    result = run_analysis(table, analysis_config_from(context.config))
    context.record(
        {
            "status": "PASS",
            "decomposition": result.paths.as_dict(),
            "mediated_fraction": result.theta.as_dict(),
            "fraction_test": theta_null_test(result.theta),
            "mediator_diagnostics": result.mediator_diagnostics,
            "attenuation": [point.as_dict() for point in result.attenuation],
            "attenuation_shape": monotone_attenuation(result.attenuation),
            "strata": [row.as_dict() for row in result.strata],
            "gradient": result.gradient.as_dict(),
            "pooled": result.pooled,
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
