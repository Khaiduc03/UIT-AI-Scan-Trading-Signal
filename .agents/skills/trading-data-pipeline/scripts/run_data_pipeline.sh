#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
make run-download
make run-validate
echo "Data pipeline completed."
