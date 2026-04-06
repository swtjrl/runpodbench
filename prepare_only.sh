#!/usr/bin/env bash
set -euo pipefail

# 1) Install deps
bash ./setup_env.sh

# 2) Generate local synthetic audio corpus
python3 ./make_audio_samples.py --out-dir ./audio_samples --durations "1,3,5,8,12,15,20"

# 3) Start vLLM in a separate shell:
#    bash ./start_vllm_gemma4.sh E2B

echo "Environment prepared. Start vLLM and then run benchmark scripts."