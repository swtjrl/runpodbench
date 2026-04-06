#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import uvicorn


@dataclass
class SessionState:
    pcm: bytearray
    sample_rate: int
    ptt_started: bool
    ptt_up_server_ts: Optional[float]


class TransformersASR:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self._torch = torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    async def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_bytes, sample_rate)

    def _transcribe_sync(self, pcm_bytes: bytes, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
            _write_wav(fp.name, pcm_bytes, sample_rate)
            out = self.pipe(fp.name)
            return out.get("text", "").strip()


class VllmASR:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
            _write_wav(fp.name, pcm_bytes, sample_rate)
            with open(fp.name, "rb") as f:
                resp = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                )
        return getattr(resp, "text", "").strip()


def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()

    static_dir = Path(__file__).parent
    uploads_dir = static_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

    asr_backend = None
    if args.asr_backend == "transformers":
        asr_backend = TransformersASR(args.whisper_model)
    else:
        asr_backend = VllmASR(args.asr_base_url, args.asr_api_key, args.asr_model)

    translator = AsyncOpenAI(base_url=args.gemma_base_url, api_key=args.gemma_api_key)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "realtime_ptt_client.html")

    @app.post("/api/e2b/audio_once")
    async def e2b_audio_once(audio: UploadFile = File(...)) -> dict:
        ts_req_start = time.perf_counter()
        safe_name = f"{int(time.time() * 1000)}_{audio.filename or 'input.wav'}"
        saved_path = uploads_dir / safe_name

        with saved_path.open("wb") as fp:
            shutil.copyfileobj(audio.file, fp)

        ts_model_start = time.perf_counter()
        audio_url = f"http://127.0.0.1:{args.port}/uploads/{safe_name}"
        prompt = f"Transcribe this audio and translate it to {args.target_lang}. Return only the translated text."
        resp = await translator.chat.completions.create(
            model=args.gemma_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=args.gemma_max_tokens,
            temperature=0.0,
        )
        ts_done = time.perf_counter()
        out_text = resp.choices[0].message.content or ""

        return {
            "ok": True,
            "file_name": safe_name,
            "translated_text": out_text,
            "elapsed": {
                "upload_plus_total_ms_server": round((ts_done - ts_req_start) * 1000.0, 2),
                "model_only_ms_server": round((ts_done - ts_model_start) * 1000.0, 2),
            },
        }

    @app.websocket("/ws/ptt")
    async def ws_ptt(ws: WebSocket) -> None:
        await ws.accept()
        state = SessionState(pcm=bytearray(), sample_rate=args.sample_rate, ptt_started=False, ptt_up_server_ts=None)
        last_partial_at = 0.0

        try:
            while True:
                msg = await ws.receive()
                if "text" in msg and msg["text"] is not None:
                    payload = json.loads(msg["text"])
                    msg_type = payload.get("type")

                    if msg_type == "start":
                        state.pcm.clear()
                        state.ptt_started = True
                        state.ptt_up_server_ts = None
                        await ws.send_text(json.dumps({"type": "ack", "phase": "start"}))

                    elif msg_type == "stop":
                        if not state.ptt_started:
                            await ws.send_text(json.dumps({"type": "error", "message": "PTT not started"}))
                            continue

                        state.ptt_up_server_ts = time.perf_counter()
                        t_asr_start = time.perf_counter()
                        asr_text = await asr_backend.transcribe_pcm(bytes(state.pcm), state.sample_rate)
                        t_asr_done = time.perf_counter()

                        prompt = (
                            f"Translate the following transcription to natural {args.target_lang}. "
                            f"Return only translated {args.target_lang} text.\n\n"
                            f"Source:\n{asr_text}"
                        )

                        t_mt_start = time.perf_counter()
                        mt_resp = await translator.chat.completions.create(
                            model=args.gemma_model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=args.gemma_max_tokens,
                            temperature=0.0,
                        )
                        translated = mt_resp.choices[0].message.content or ""
                        t_mt_done = time.perf_counter()

                        elapsed_asr_ms = (t_asr_done - t_asr_start) * 1000.0
                        elapsed_mt_ms = (t_mt_done - t_mt_start) * 1000.0
                        elapsed_from_ptt_up_ms = (t_mt_done - state.ptt_up_server_ts) * 1000.0

                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "final",
                                    "asr_text": asr_text,
                                    "translated_text": translated,
                                    "elapsed": {
                                        "asr_ms": round(elapsed_asr_ms, 2),
                                        "mt_ms": round(elapsed_mt_ms, 2),
                                        "ptt_up_to_translated_ms_server": round(elapsed_from_ptt_up_ms, 2),
                                    },
                                },
                                ensure_ascii=False,
                            )
                        )

                        state.ptt_started = False

                    elif msg_type == "ping":
                        await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

                elif "bytes" in msg and msg["bytes"] is not None:
                    if not state.ptt_started:
                        continue
                    chunk = msg["bytes"]
                    state.pcm.extend(chunk)

                    if args.enable_partials:
                        now = time.perf_counter()
                        if (now - last_partial_at) >= args.partial_interval_sec and len(state.pcm) >= args.sample_rate * 2:
                            last_partial_at = now
                            partial = await asr_backend.transcribe_pcm(bytes(state.pcm), state.sample_rate)
                            await ws.send_text(json.dumps({"type": "partial", "text": partial}, ensure_ascii=False))

        except WebSocketDisconnect:
            return

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime PTT benchmark: whisper-small-komix -> gemma4 translation")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--sample-rate", default=16000, type=int)

    parser.add_argument("--asr-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--whisper-model", default="seastar105/whisper-small-komixv2", type=str)

    parser.add_argument("--asr-base-url", default="http://127.0.0.1:8001/v1", type=str)
    parser.add_argument("--asr-api-key", default="EMPTY", type=str)
    parser.add_argument("--asr-model", default="seastar105/whisper-small-komixv2", type=str)

    parser.add_argument("--gemma-base-url", default="http://127.0.0.1:8000/v1", type=str)
    parser.add_argument("--gemma-api-key", default="EMPTY", type=str)
    parser.add_argument("--gemma-model", default="google/gemma-4-E2B-it", type=str)
    parser.add_argument("--gemma-max-tokens", default=128, type=int)
    parser.add_argument("--target-lang", default="Japanese", type=str)

    parser.add_argument("--enable-partials", action="store_true")
    parser.add_argument("--partial-interval-sec", default=1.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
