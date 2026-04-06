#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install fastapi uvicorn requests python-multipart "transformers>=4.50.0" accelerate torch soundfile librosa

echo "Starting realtime PTT server (Transformers-only: whisper-small-komixv2 + gemma-4-E2B-it)..."
python3 ./realtime_ptt_server.py \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-model google/gemma-4-E2B-it \
  --target-lang Japanese \
  --port 9000