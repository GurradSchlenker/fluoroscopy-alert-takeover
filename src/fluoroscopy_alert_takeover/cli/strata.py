"""Tabulate the per-condition breakdown and the multi-centre heterogeneity.

Ref: Table 3, pp. 9-10; Table A2, p. 26; Sec. 4.7, p. 19.
"""

from __future__ import annotations

import numpy as np

from ..analysis.pipeline import panel_sizes, run_analysis
from ..strata.heterogeneity import heterogeneity_report
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Record the stratum rows, the pooled rows and the between-site heterogeneity."""
    context = prepare_context("strata", "Tabulate the per-condition breakdown", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    result = run_analysis(table, analysis_config_from(context.config))
    site_rows = [row for row in result.strata if row.group == "site"]
    if len(site_rows) < 2:
        context.record(
            {"status": "FAIL", "reason": "fewer than two site strata in the certified region"}
        )
        return EXIT_OK
    points = np.asarray([row.takeover for row in site_rows], dtype=np.float64)
    errors = np.asarray([row.takeover_se for row in site_rows], dtype=np.float64)
    if bool((errors <= 0.0).any()):
        context.record({"status": "FAIL", "reason": "a site stratum carries no standard error"})
        return EXIT_OK
    context.record(
        {
            "status": "PASS",
            "strata": [row.as_dict() for row in result.strata],
            "pooled": result.pooled,
            "gradient": result.gradient.as_dict(),
            "subgroup_gap": result.subgroup_gap,
            "panel_sizes": panel_sizes(table),
            "site_points_pp": [float(value) for value in points],
            "heterogeneity": heterogeneity_report(points, errors).as_dict(),
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
