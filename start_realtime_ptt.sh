#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install fastapi uvicorn openai "transformers>=4.50.0" accelerate torch soundfile librosa

echo "Starting realtime PTT server (transformers backend with whisper-small-komixv2)..."
python3 ./realtime_ptt_server.py \
  --asr-backend transformers \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-base-url http://127.0.0.1:8000/v1 \
  --gemma-model google/gemma-4-E2B-it \
  --enable-partials