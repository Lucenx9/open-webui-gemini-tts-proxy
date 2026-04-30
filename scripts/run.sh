#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${APP_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${APP_DIR}/.env"
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required. Set it in the environment or in .env." >&2
  exit 1
fi

exec "${APP_DIR}/.venv/bin/uvicorn" app:app \
  --host "${GEMINI_TTS_PROXY_HOST:-0.0.0.0}" \
  --port "${GEMINI_TTS_PROXY_PORT:-18792}" \
  --proxy-headers
