#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

TARGET_LANG="${TARGET_LANG:-Japanese}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-E2B-it}"
GEMMA_PORT="${GEMMA_PORT:-8000}"
PTT_PORT="${PTT_PORT:-9000}"

echo "[1/5] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install "vllm[audio]" openai fastapi uvicorn requests python-multipart "transformers>=4.50.0" accelerate torch soundfile librosa

echo "[2/5] Starting Gemma4 E2B on port ${GEMMA_PORT}..."
nohup python3 -m vllm serve "${GEMMA_MODEL}" \
  --host 0.0.0.0 \
  --port "${GEMMA_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt image=4,audio=1 \
  --async-scheduling \
  > "${ROOT_DIR}/logs_gemma.txt" 2>&1 &

echo "[3/5] Waiting for Gemma server..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${GEMMA_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${GEMMA_PORT}/v1/models" >/dev/null; then
  echo "Gemma server did not become ready. Check logs_gemma.txt"
  exit 1
fi

echo "[4/5] Starting realtime PTT server on port ${PTT_PORT}..."
nohup python3 "${ROOT_DIR}/realtime_ptt_server.py" \
  --asr-backend transformers \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-base-url "http://127.0.0.1:${GEMMA_PORT}/v1" \
  --gemma-model "${GEMMA_MODEL}" \
  --target-lang "${TARGET_LANG}" \
  --port "${PTT_PORT}" \
  --enable-partials \
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

echo "[5/5] Creating public URL without port/SSH setup..."
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "${ROOT_DIR}/cloudflared"
  chmod +x "${ROOT_DIR}/cloudflared"
  CF_BIN="${ROOT_DIR}/cloudflared"
else
  CF_BIN="$(command -v cloudflared)"
fi

nohup "${CF_BIN}" tunnel --url "http://127.0.0.1:${PTT_PORT}" > "${ROOT_DIR}/logs_tunnel.txt" 2>&1 &
sleep 5

URL="$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${ROOT_DIR}/logs_tunnel.txt" | head -n 1 || true)"

echo
echo "===== READY ====="
echo "PTT Web URL: ${URL:-not-found-yet}"
echo "If URL is empty, run: tail -f ${ROOT_DIR}/logs_tunnel.txt"
echo "Gemma log: ${ROOT_DIR}/logs_gemma.txt"
echo "PTT log:   ${ROOT_DIR}/logs_ptt.txt"

