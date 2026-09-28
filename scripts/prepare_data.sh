#!/usr/bin/env bash
# Check that a cohort root satisfies the release layout.
#
# The record-level cohort is held under site data-sharing agreements and is never shipped,
# so this script only inspects whatever root is pointed at. It never writes into it.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COHORT="${2:-${COHORT_ROOT:-data/cohort}}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "checking cohort root: ${COHORT}"
"${PYTHON_BIN}" - "${COHORT}" <<'PY'
import sys
from pathlib import Path

from fluoroscopy_alert_takeover.cohort.layout import CohortLayout

root = Path(sys.argv[1])
layout = CohortLayout(root=root)
missing = layout.missing()
if missing:
    print("missing entries: " + ", ".join(missing))
    print("the cohort root is not staged; every command that needs it will report BLOCKED")
    raise SystemExit(2)
procedures = layout.procedure_ids()
print(f"navigation logs present: {len(procedures)}")
print(f"site registry: {layout.site_registry}")
print(f"procedure table: {layout.procedures}")
print(f"reader layer: {'present' if layout.reader_study.is_file() else 'absent'}")
PY
