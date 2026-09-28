"""Run the pre-specified falsification pair of Algorithm 2.

Ref: Algorithm 2, p. 16; Table 4 Panel A and Panel C, pp. 11-12.
"""

from __future__ import annotations

from ..analysis.pipeline import run_analysis
from .common import EXIT_BLOCKED, EXIT_OK, analysis_config_from, load_table, prepare_context


def main(argv: list[str] | None = None) -> int:
    """Record the negative-control panel, the exposure-side probe and the sensitivity grid."""
    context = prepare_context("falsify", "Run the falsification pair", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    result = run_analysis(table, analysis_config_from(context.config))
    report = result.falsification
    context.record(
        {"status": report.verdict, **report.as_dict(), "probes": [vars(p) for p in result.probes]}
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
