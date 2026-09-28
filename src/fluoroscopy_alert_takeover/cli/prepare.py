"""Build the person-window table from a cohort root.

Ref: Sec. 4.5, pp. 18-19 - the two data layers and their linkage.
"""

from __future__ import annotations

from ..cohort.builder import cohort_digest
from .common import (
    EXIT_BLOCKED,
    EXIT_OK,
    cohort_root_from,
    load_table,
    prepare_context,
    window_spec_from,
)


def main(argv: list[str] | None = None) -> int:
    """Read the cohort layers and record the person-window table digest."""
    context = prepare_context("prepare", "Assemble the analytic person-window table", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record(
            {
                "status": "BLOCKED",
                "reason": str(error),
                "root": str(cohort_root_from(context.config)),
            }
        )
        return EXIT_BLOCKED
    context.record(
        {"status": "PASS", "spec": vars(window_spec_from(context.config)), **cohort_digest(table)}
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
