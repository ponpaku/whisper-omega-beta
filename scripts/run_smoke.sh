#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

.venv-system/bin/python scripts/create_smoke_wav.py
.venv-system/bin/omega doctor --json-output
.venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --output-format json --emit-result-json always

