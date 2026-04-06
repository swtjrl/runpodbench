# runpodbench

## 진짜 한 번에 실행

Runpod Web Terminal에서 아래 3줄만 실행:

```bash
cd /workspace
git clone https://github.com/swtjrl/runpodbench.git || true
cd /workspace/runpodbench && git pull && chmod +x *.sh && bash ./easy_start_no_jupyter.sh
```

끝나면 터미널에 `PTT Web URL`이 출력된다.
그 URL을 네 PC 브라우저에서 열고 `Connect WS` -> PTT 테스트.

## 로그 확인(문제 있을 때)

```bash
tail -n 120 /workspace/runpodbench/logs_gemma.txt
tail -n 120 /workspace/runpodbench/logs_ptt.txt
tail -n 120 /workspace/runpodbench/logs_tunnel.txt
```