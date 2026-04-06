#!/usr/bin/env bash
set -euo pipefail

# 1) Install deps
bash ./setup_env.sh

# 2) Generate local synthetic audio corpus
python3 ./make_audio_samples.py --out-dir ./audio_samples --durations "1,3,5,8,12,15,20"

echo "Environment prepared. Run start_realtime_ptt.sh or easy_start_no_jupyter.sh"