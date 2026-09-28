#!/usr/bin/env bash
# Train the navigation alert policy.
#
# The policy is trained where the cine frames are staged; the container that runs the
# offline analysis carries no accelerator runtime, because the manuscript's own offline
# figure (18.4 min per 10,000 procedures on 8 CPU cores) is a CPU figure.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${2:-configs/experiment/primary.yaml}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m fluoroscopy_alert_takeover.cli.train_alert --experiment "${EXPERIMENT}" "$@"
