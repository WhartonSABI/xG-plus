#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

cmd=(
  "$PYTHON_BIN" scripts/run_pipeline.py
  --competition pl \
  --seasons 2022-2023 2023-2024 2024-2025 \
  --workers 12 \
  --repair-corrupt-tracking \
  --continue-on-unrepairable-corrupt-tracking \
  --allow-temp-files \
  --preprocess-only
)

if [[ -n "${TRACKING_CREDENTIALS_FROM:-}" ]]; then
  cmd+=(--tracking-credentials-from "$TRACKING_CREDENTIALS_FROM")
elif [[ -f archived/local/sagemaker/features.py ]]; then
  cmd+=(--tracking-credentials-from archived/local/sagemaker/features.py)
fi

cmd+=("$@")

if [[ "${CAFFEINATE:-1}" != "0" ]] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dims "${cmd[@]}"
fi

exec "${cmd[@]}"
