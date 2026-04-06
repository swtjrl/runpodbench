#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

TARGET_LANG="${TARGET_LANG:-Japanese}"
PTT_PORT="${PTT_PORT:-9000}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

pkill -f "realtime_ptt_server.py" >/dev/null 2>&1 || true
pkill -f "cloudflared tunnel --url http://127.0.0.1:${PTT_PORT}" >/dev/null 2>&1 || true

echo "[1/4] Creating venv and installing dependencies (no vLLM)..."
python3 -m venv "${VENV_DIR}"
VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

"${VENV_PIP}" install --upgrade pip
"${VENV_PIP}" install --upgrade --no-cache-dir \
  fastapi uvicorn requests python-multipart \
  "transformers>=4.50.0" accelerate torch soundfile librosa packaging \
  git+https://github.com/huggingface/transformers.git

echo "[2/4] Verifying gemma4 support..."
"${VENV_PY}" - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
print("model_type:", cfg.model_type)
assert cfg.model_type == "gemma4"
PY

echo "[3/4] Starting unified realtime server (Whisper + Gemma via Transformers)..."
nohup "${VENV_PY}" "${ROOT_DIR}/realtime_ptt_server.py" \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-model google/gemma-4-E2B-it \
  --target-lang "${TARGET_LANG}" \
  --port "${PTT_PORT}" \
  > "${ROOT_DIR}/logs_ptt.txt" 2>&1 &

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PTT_PORT}/" >/dev/null; then
    break
  fi
  if (( i % 10 == 0 )); then
    echo "[3/4] still waiting... (${i}/240)"
    tail -n 12 "${ROOT_DIR}/logs_ptt.txt" || true
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PTT_PORT}/" >/dev/null; then
  echo "PTT server did not become ready. Check logs_ptt.txt"
  exit 1
fi

echo "[4/4] Creating public URL..."
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
echo "PTT log:   ${ROOT_DIR}/logs_ptt.txt"
echo "Tunnel log:${ROOT_DIR}/logs_tunnel.txt"