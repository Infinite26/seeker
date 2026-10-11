#!/usr/bin/env bash
# Start Seeker (voice wake assistant).
#   ./run.sh                 # run in foreground, Ctrl-C to stop
#   ./run.sh --start-tts     # also start the Kokoro TTS sidecar if it is down
#   ./run.sh --list-devices  # show ALSA capture devices
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--list-devices" ]; then
  arecord -l
  exit 0
fi

exec ./.venv/bin/python -m seeker_voice "$@"
