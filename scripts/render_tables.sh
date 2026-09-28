#!/usr/bin/env bash
# Render the reported tables from one analysis run.
#
# The tables command recomposes the analysis rather than reading the stage records, so the
# numbers it writes are always produced by the configuration it was given.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${2:-configs/experiment/primary.yaml}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m fluoroscopy_alert_takeover.cli.tables --experiment "${EXPERIMENT}" "$@"
