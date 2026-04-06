# runpodbench (no vLLM)

## One command

Runpod Web Terminal:

```bash
cd /workspace
git clone https://github.com/swtjrl/runpodbench.git || true
cd /workspace/runpodbench && git pull && chmod +x *.sh && bash ./easy_start_no_jupyter.sh
```

This now runs **Transformers-only** pipeline:
- ASR: `seastar105/whisper-small-komixv2`
- MT: `google/gemma-4-E2B-it`

After startup, open `PTT Web URL` and test.

## Logs

```bash
tail -n 120 /workspace/runpodbench/logs_ptt.txt
tail -n 120 /workspace/runpodbench/logs_tunnel.txt
```