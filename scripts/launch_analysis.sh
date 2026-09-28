#!/usr/bin/env bash
# Run the pre-specified analysis end to end.
#
# The stages are separate commands on purpose: each writes its own record under the run
# directory, so a stage that blocks names itself instead of failing the whole run.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${2:-configs/experiment/primary.yaml}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for stage in prepare gate estimate decompose falsify strata operating_point tables; do
  echo "--- ${stage}"
  "${PYTHON_BIN}" -m "fluoroscopy_alert_takeover.cli.${stage}" --experiment "${EXPERIMENT}" "$@"
done

echo "records written under runs/"
