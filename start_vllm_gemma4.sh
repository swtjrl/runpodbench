#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-E2B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/workspace/runpodbench/uploads}"

if [[ "${MODEL_SIZE}" == "E2B" ]]; then
  MODEL_ID="google/gemma-4-E2B-it"
elif [[ "${MODEL_SIZE}" == "E4B" ]]; then
  MODEL_ID="google/gemma-4-E4B-it"
else
  echo "Usage: $0 [E2B|E4B]"
  exit 1
fi

echo "Launching ${MODEL_ID} on ${HOST}:${PORT}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-100}"
export VLLM_AUDIO_FETCH_TIMEOUT="${VLLM_AUDIO_FETCH_TIMEOUT:-60}"

python3 -m vllm serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --limit-mm-per-prompt image=4,audio=1 \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --async-scheduling
