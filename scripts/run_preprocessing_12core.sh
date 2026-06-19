#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

cmd=(
  "$PYTHON_BIN" scripts/run_pipeline.py
  --competition pl \
  --seasons 2022-2023 2023-2024 2024-2025 \
  --workers 12 \
  --preprocess-only
  "$@"
)

if [[ "${CAFFEINATE:-1}" != "0" ]] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dims "${cmd[@]}"
fi

exec "${cmd[@]}"
