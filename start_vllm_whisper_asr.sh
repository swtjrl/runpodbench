#!/usr/bin/env bash
set -euo pipefail

# Serve whisper-small-komixv2 with vLLM on port 8001 for /v1/audio/transcriptions.
MODEL="${MODEL:-seastar105/whisper-small-komixv2}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"

python3 -m vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len 448 \
  --gpu-memory-utilization 0.9
