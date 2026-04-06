#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

TARGET_LANG="${TARGET_LANG:-Japanese}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-E2B-it}"
GEMMA_PORT="${GEMMA_PORT:-8000}"
PTT_PORT="${PTT_PORT:-9000}"
UPLOADS_DIR="${UPLOADS_DIR:-${ROOT_DIR}/uploads}"
VLLM_MIN_VERSION="${VLLM_MIN_VERSION:-0.16.0}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

mkdir -p "${UPLOADS_DIR}"

# Clean old processes for one-click restart
pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
pkill -f "vllm serve ${GEMMA_MODEL}" >/dev/null 2>&1 || true
pkill -f "realtime_ptt_server.py" >/dev/null 2>&1 || true
pkill -f "cloudflared tunnel --url http://127.0.0.1:${PTT_PORT}" >/dev/null 2>&1 || true

echo "[1/5] Installing dependencies..."
python3 -m venv "${VENV_DIR}"
VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

"${VENV_PIP}" install --upgrade pip
"${VENV_PIP}" install "vllm[audio]" openai fastapi uvicorn requests python-multipart "transformers>=4.50.0" accelerate torch soundfile librosa packaging
"${VENV_PIP}" install --upgrade --no-cache-dir --force-reinstall git+https://github.com/huggingface/transformers.git

echo "[1.5/5] Checking vLLM version..."
"${VENV_PY}" - <<PY
import sys
from packaging.version import Version
import vllm

current = Version(vllm.__version__)
minimum = Version("${VLLM_MIN_VERSION}")
if current < minimum:
    print(f"vLLM too old: {current} < {minimum}")
    sys.exit(1)
print(f"vLLM version OK: {current}")
PY

echo "[1.6/5] Checking Transformers supports gemma4..."
"${VENV_PY}" - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
print("model_type:", cfg.model_type)
assert cfg.model_type == "gemma4"
PY

echo "[2/5] Starting Gemma4 E2B on port ${GEMMA_PORT}..."
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-100}"
export VLLM_AUDIO_FETCH_TIMEOUT="${VLLM_AUDIO_FETCH_TIMEOUT:-60}"

nohup "${VENV_PY}" -m vllm.entrypoints.openai.api_server \
  --model "${GEMMA_MODEL}" \
  --host 0.0.0.0 \
  --port "${GEMMA_PORT}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":4,"audio":1}' \
  --allowed-local-media-path "${UPLOADS_DIR}" \
  --async-scheduling \
  > "${ROOT_DIR}/logs_gemma.txt" 2>&1 &

echo "[3/5] Waiting for Gemma server..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${GEMMA_PORT}/v1/models" >/dev/null; then
    break
  fi
  if (( i % 5 == 0 )); then
    echo "[3/5] still waiting... (${i}/180)"
    echo "----- gemma log tail -----"
    tail -n 12 "${ROOT_DIR}/logs_gemma.txt" || true
    echo "----- gpu -----"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    echo "--------------------------"
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${GEMMA_PORT}/v1/models" >/dev/null; then
  echo "Gemma server did not become ready. Check logs_gemma.txt"
  exit 1
fi

echo "[3.5/5] Verifying Gemma completion..."
CHECK_PAYLOAD="$(cat <<JSON
{
  "model": "${GEMMA_MODEL}",
  "messages": [{"role":"user","content":"Reply with exactly: ok"}],
  "max_tokens": 8,
  "temperature": 0
}
JSON
)"
CHECK_RESP="$(curl -sS "http://127.0.0.1:${GEMMA_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${CHECK_PAYLOAD}" || true)"

if ! echo "${CHECK_RESP}" | grep -q '"choices"'; then
  echo "Gemma endpoint is up but completion failed."
  echo "Check logs: tail -n 200 ${ROOT_DIR}/logs_gemma.txt"
  exit 1
fi

echo "[4/5] Starting realtime PTT server on port ${PTT_PORT}..."
nohup "${VENV_PY}" "${ROOT_DIR}/realtime_ptt_server.py" \
  --asr-backend transformers \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-base-url "http://127.0.0.1:${GEMMA_PORT}/v1" \
  --gemma-model "${GEMMA_MODEL}" \
  --target-lang "${TARGET_LANG}" \
  --port "${PTT_PORT}" \
  > "${ROOT_DIR}/logs_ptt.txt" 2>&1 &

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PTT_PORT}/" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:${PTT_PORT}/" >/dev/null; then
  echo "PTT server did not become ready. Check logs_ptt.txt"
  exit 1
fi

echo "[5/5] Creating public URL..."
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "${ROOT_DIR}/cloudflared"
  chmod +x "${ROOT_DIR}/cloudflared"
  CF_BIN="${ROOT_DIR}/cloudflared"
else
  CF_BIN="$(command -v cloudflared)"
fi

nohup "${CF_BIN}" tunnel --url "http://127.0.0.1:${PTT_PORT}" --protocol http2 > "${ROOT_DIR}/logs_tunnel.txt" 2>&1 &
sleep 6

URL="$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${ROOT_DIR}/logs_tunnel.txt" | head -n 1 || true)"
echo "${URL}" > "${ROOT_DIR}/PTT_URL.txt"

echo
echo "===== READY ====="
echo "PTT Web URL: ${URL:-not-found-yet}"
echo "Saved URL: ${ROOT_DIR}/PTT_URL.txt"
echo "Gemma log: ${ROOT_DIR}/logs_gemma.txt"
echo "PTT log:   ${ROOT_DIR}/logs_ptt.txt"
echo "Tunnel log:${ROOT_DIR}/logs_tunnel.txt"
