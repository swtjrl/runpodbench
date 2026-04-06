# Runpod Gemma4 + Whisper Benchmark Kit

이 폴더는 Runpod GPU를 켠 뒤 바로 측정할 수 있도록, 사전 준비용 스크립트를 모아둔 것입니다.

## Files

- `setup_env.sh`: 벤치에 필요한 Python 패키지 설치
- `start_vllm_gemma4.sh`: Gemma4 E2B/E4B 서버 실행
- `make_audio_samples.py`: 길이별 테스트용 WAV 샘플 생성
- `bench_gemma4_audio.py`: Gemma4 오디오 단건 지연 + 배치 스트레스 테스트
- `bench_whisper_komix_to_gemma.py`: Whisper small komix -> Gemma4 E2B 번역 지연 측정
- `realtime_ptt_server.py`: 웹소켓 PTT 서버(ASR + Gemma 번역 + elapsed 측정)
- `realtime_ptt_client.html`: 브라우저 PTT 테스트 UI
- `start_realtime_ptt.sh`: small-komix(Transformers) 실시간 테스트 서버 실행
- `prepare_only.sh`: 환경 준비만 한 번에 실행

## Quick Start (Runpod/JupyterLab Terminal)

```bash
cd /workspace/82hz/runpod_bench
chmod +x *.sh
bash ./prepare_only.sh
```

### 1) Gemma4 서버 실행

```bash
# E2B
bash ./start_vllm_gemma4.sh E2B
```

기본 주소는 `http://127.0.0.1:8000/v1` 입니다.

## Realtime PTT Test (네가 요청한 테스트)

목표: `PTT에서 손 뗀 순간`부터 `Gemma4 번역 텍스트 도착`까지 elapsed 측정

### A. Whisper small-komix 로컬 로드(기본)

```bash
bash ./start_realtime_ptt.sh
```

브라우저에서 `http://<RUNPOD_IP>:9000` 접속 후:
- `Connect WS`
- PTT 버튼 누르고 말하기
- 손 떼면 final ASR/번역 + elapsed 표시

표시되는 elapsed:
- `asr_ms`
- `mt_ms`
- `ptt_up_to_translated_ms_server`
- `ptt_up_to_translated_ms_client`

### B. vLLM을 ASR로 쓰고 싶을 때 (옵션)

1) 별도 포트(예: 8001)에서 Whisper 계열 vLLM 서버 실행 (ash ./start_vllm_whisper_asr.sh)
2) PTT 서버를 `--asr-backend vllm`으로 실행

```bash
python3 ./realtime_ptt_server.py \
  --asr-backend vllm \
  --asr-base-url http://127.0.0.1:8001/v1 \
  --asr-model seastar105/whisper-small-komixv2 \
  --gemma-base-url http://127.0.0.1:8000/v1 \
  --gemma-model google/gemma-4-E2B-it \
  --enable-partials
```

참고: vLLM ASR 백엔드는 `/v1/audio/transcriptions` 경로를 사용한다.

## Existing batch tests

### Gemma4 오디오 단건 + 배치 스트레스

```bash
python3 ./bench_gemma4_audio.py \
  --model google/gemma-4-E2B-it \
  --audio-dir ./audio_samples \
  --single-rounds 5 \
  --batch-rounds 20 \
  --batch-size 16 \
  --concurrency 16 \
  --csv ./results/gemma4_audio_e2b.csv
```

### Whisper small komix -> Gemma4 번역 지연

```bash
python3 ./bench_whisper_komix_to_gemma.py \
  --audio-dir ./audio_samples \
  --whisper-model seastar105/whisper-small-komixv2 \
  --gemma-model google/gemma-4-E2B-it \
  --rounds 3
```

