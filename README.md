# Runpod Gemma4 + Whisper Bench

## 가장 쉬운 방법 (Jupyter/SSH 포트 설정 없이)

Runpod `Connect` 탭에서 `Enable web terminal` 켠 뒤, 웹 터미널에서:

```bash
cd /workspace/runpodbench
chmod +x *.sh
bash ./easy_start_no_jupyter.sh
```

실행이 끝나면 `PTT Web URL`(trycloudflare 주소)이 출력된다.
그 URL을 네 컴퓨터 브라우저에서 열어서 바로 테스트하면 된다.
## 목표

1. `small-komix` PTT 실시간: 한국어 발화 -> 타겟 언어 번역까지 elapsed
2. `e2b` 오디오 단건(비실시간): 로컬PC 업로드 -> 결과 수신 elapsed
3. `e2b` 오디오 스트레스: 배치/동시 요청 지연

## 0) Runpod에서 준비

```bash
cd /workspace
git clone https://github.com/swtjrl/runpodbench.git
cd runpodbench
chmod +x *.sh
```

## 1) Gemma4 E2B 서버 (Runpod)

```bash
bash ./start_vllm_gemma4.sh E2B
```

## 2) PTT + 업로드 API 서버 (Runpod)

별도 터미널에서:

```bash
bash ./start_realtime_ptt.sh
```

이 서버가 제공하는 것:
- 웹 PTT: `http://<RUNPOD_IP>:9000`
- 업로드 API: `POST http://<RUNPOD_IP>:9000/api/e2b/audio_once`

## 목표 1: 웹 UI PTT 실시간 측정

브라우저에서:
- `http://<RUNPOD_IP>:9000` 접속
- `Connect WS`
- PTT 누르고 말함
- 손 떼면 elapsed 출력

지표:
- `asr_ms`
- `mt_ms`
- `ptt_up_to_translated_ms_server`
- `ptt_up_to_translated_ms_client`

## 목표 2/3: 로컬PC -> Runpod 업로드 지연 측정

로컬PC에서 저장소 같은 폴더 열고:

```bash
python3 ./client_e2b_upload_bench.py \
  --base-url http://<RUNPOD_IP>:9000 \
  --audio-dir ./audio_samples \
  --single-rounds 5 \
  --stress-total-requests 64 \
  --stress-concurrency 8
```

`--stress-concurrency 16`으로 바꾸면 batch 16급 부하 테스트 가능.

## WAV/MP3

- PTT 웹 테스트: 파일 필요 없음(마이크 직접 입력)
- 업로드 벤치: 기본은 `WAV` 권장
- 샘플 생성:

```bash
python3 ./make_audio_samples.py --out-dir ./audio_samples --durations "1,3,5,8,12,15"
```

## vLLM ASR를 쓰고 싶을 때 (옵션)

```bash
bash ./start_vllm_whisper_asr.sh

python3 ./realtime_ptt_server.py \
  --asr-backend vllm \
  --asr-base-url http://127.0.0.1:8001/v1 \
  --asr-model seastar105/whisper-small-komixv2 \
  --gemma-base-url http://127.0.0.1:8000/v1 \
  --gemma-model google/gemma-4-E2B-it \
  --target-lang Japanese \
  --enable-partials
```

### e2b가 실제 동작하는지 확인

`easy_start_no_jupyter.sh`는 이제 `/v1/chat/completions` 실응답까지 확인한다.
실패하면 즉시 중단하고 `logs_gemma.txt` 확인 안내를 출력한다.

