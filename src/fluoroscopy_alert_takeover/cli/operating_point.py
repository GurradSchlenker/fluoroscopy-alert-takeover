"""Select the operating point over the abstention band.

Ref: Algorithm 5, p. 20; Table A3 Panel A and Panel B, p. 27; Table 4 Panel D, p. 12.
"""

from __future__ import annotations

from ..analysis.pipeline import contrast_pair
from ..operating_point.abstention import abstention_band_comparison, select_operating_point
from ..operating_point.benefit_curve import causal_benefit_curve, interrupt_burden, peak_threshold
from ..utils.config import dig
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Sweep the threshold, freeze the operating point and record the band comparison."""
    context = prepare_context("operating_point", "Select the alert operating point", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    config = analysis_config_from(context.config)
    curve = causal_benefit_curve(table, table.alert_score, contrast_pair(config), config.thresholds)
    point = select_operating_point(
        curve,
        burden_ceiling=float(dig(context.config, "operating_point.burden_ceiling", 0.60)),
        utility_floor=float(dig(context.config, "operating_point.utility_floor", 0.0)),
    )
    context.record(
        {
            "status": "PASS",
            "curve": curve.as_dict(),
            "peak_threshold_takeover": peak_threshold(curve, "takeover"),
            "operating_point": point.as_dict(),
            "interrupts_per_procedure": interrupt_burden(table, table.alert | ~table.alert),
            "band_comparison": abstention_band_comparison(
                table, table.alert_score, table.alert, config.abstention, point.threshold
            ),
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
