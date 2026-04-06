#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

${PYTHON_BIN} -m pip install --upgrade pip
${PYTHON_BIN} -m pip install fastapi uvicorn requests python-multipart "transformers>=4.50.0" accelerate torch soundfile librosa huggingface_hub

echo "Done. Optional next step: huggingface-cli login"