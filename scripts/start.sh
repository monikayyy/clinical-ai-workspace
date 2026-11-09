#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONDONTWRITEBYTECODE=1
# Single worker is mandatory. HTTP is permitted only on loopback for this prototype.
# Disable core dumps and HTTP access logs; never enable debug logging with patient data.
ulimit -c 0 || true
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --log-level critical
