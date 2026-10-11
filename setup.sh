#!/usr/bin/env bash
# One-time setup for Seeker: venv + Python deps.
set -euo pipefail
cd "$(dirname "$0")"

uv venv .venv --python 3.11
uv pip install --python ./.venv/bin/python faster-whisper numpy requests

echo
echo "Seeker venv ready. Start with ./run.sh"
