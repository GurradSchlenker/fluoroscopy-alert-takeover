"""Render the reported tables into one record.

Ref: Table 1, pp. 6-7; Table 2, p. 8; Table 3, pp. 9-10; Table 4, pp. 11-12; Table A1,
p. 25; Table A2, p. 26; Table A3, p. 27.
"""

from __future__ import annotations

from ..analysis.pipeline import run_analysis
from ..cohort.layout import CohortLayout
from ..evaluation.discrimination import discrimination_report, evaluate
from ..evaluation.reader_study import ArmCounts, load_reader_arms
from ..operating_point.abstention import band_scores
from ..workflow.latency import StageLedger
from ..workflow.procedure_metrics import describe, integration_cost
from .common import (
    EXIT_BLOCKED,
    EXIT_OK,
    analysis_config_from,
    cohort_root_from,
    load_table,
    prepare_context,
)


def main(argv: list[str] | None = None) -> int:
    """Assemble every reported table from one analysis run."""
    context = prepare_context("tables", "Render the reported tables", argv)
    try:
        table = load_table(context.config)
    except FileNotFoundError as error:
        context.record({"status": "BLOCKED", "reason": str(error)})
        return EXIT_BLOCKED
    config = analysis_config_from(context.config)
    layout = CohortLayout(root=cohort_root_from(context.config))
    arms: list[ArmCounts] | None = None
    if layout.reader_study.is_file():
        arms = load_reader_arms(layout.reader_study)
    # The three-arm reader comparison is a separate observational layer, so it is passed in
    # rather than reconstructed from the person-window table.
    result = run_analysis(table, config, reader_arms=arms)

    banded = band_scores(table.alert_score, config.abstention)
    ledger = StageLedger()
    context.record(
        {
            "status": "PASS",
            "table1": {
                "contrasts": result.table_record,
                "alert_model": discrimination_report(
                    table.alert_score, table.alert, seed=config.seed
                ),
                "reader_comparison": result.reader,
            },
            "table2": {
                "rows": [row.as_dict() for row in result.ablation],
                "summary": result.ablation_summary,
            },
            "table3": {
                "strata": [row.as_dict() for row in result.strata],
                "pooled": result.pooled,
                "gradient": result.gradient.as_dict(),
                "subgroup_gap": result.subgroup_gap,
            },
            "table4": {
                "falsification": result.falsification.as_dict(),
                "attenuation": [point.as_dict() for point in result.attenuation],
                "alternatives": result.alternatives,
                "band": result.band_comparison,
                "band_auroc": {
                    "unrestricted": evaluate(table.alert_score, table.alert).as_dict(),
                    "banded": evaluate(banded, table.alert).as_dict(),
                },
            },
            "table_a1": ledger.latencies(),
            "table_a3": {"curve": result.benefit_curve.as_dict()},
            "success": result.success,
            "workflow": {
                "fluoroscopy_time_min": describe(table, "fluoroscopy_time_min").as_dict(),
                "contrast_volume_ml": describe(table, "contrast_volume_ml").as_dict(),
                "procedure_duration_min": describe(table, "procedure_duration_min").as_dict(),
                "integration_cost": integration_cost(table),
            },
            "multiplicity": result.family_multiplicity,
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
