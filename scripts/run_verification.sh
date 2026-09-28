#!/usr/bin/env bash
# Run both verification layers and refresh the three artefacts.
#
# Without a staged cohort the execution layer records the cohort-dependent checks as
# BLOCKED and still writes the artefacts, so the report always reflects what was actually
# run rather than what was intended.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${2:-configs/experiment/primary.yaml}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m fluoroscopy_alert_takeover.cli.verify --experiment "${EXPERIMENT}"

echo "artefacts refreshed: claim_to_code.json, verification_report.json, integrity_manifest.json"
