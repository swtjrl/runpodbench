#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _read_audio_to_pcm16(path: Path, target_sr: int = 16000) -> tuple[bytes, int]:
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    return pcm, target_sr


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
        wav = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        out = self.pipe({"array": wav, "sampling_rate": sample_rate})
        return out.get("text", "").strip()


class GemmaTranslator:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import pipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            task="text-generation",
            model=model_id,
            torch_dtype=dtype,
            device=device,
        )

    async def translate(self, text: str, target_lang: str, max_new_tokens: int) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._translate_sync, text, target_lang, max_new_tokens)

    def _translate_sync(self, text: str, target_lang: str, max_new_tokens: int) -> str:
        prompt = (
            f"Translate the following text to natural {target_lang}. "
            f"Return only the translated {target_lang} text.\n\n"
            f"Source:\n{text}"
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
        return (outputs[0].get("generated_text", "") or "").strip()


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()

    static_dir = Path(__file__).parent
    uploads_dir = static_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    asr_backend = TransformersASR(args.whisper_model)
    translator = GemmaTranslator(args.gemma_model)

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

        try:
            pcm, sr = _read_audio_to_pcm16(saved_path, target_sr=args.sample_rate)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "file_name": safe_name, "error": f"audio decode failed: {exc}"}

        ts_model_start = time.perf_counter()
        try:
            asr_text = await asr_backend.transcribe_pcm(pcm, sr)
            translated = await translator.translate(asr_text, args.target_lang, args.gemma_max_tokens)
            ts_done = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            ts_done = time.perf_counter()
            return {
                "ok": False,
                "file_name": safe_name,
                "error": str(exc),
                "elapsed": {
                    "upload_plus_total_ms_server": round((ts_done - ts_req_start) * 1000.0, 2),
                    "model_only_ms_server": round((ts_done - ts_model_start) * 1000.0, 2),
                },
            }

        return {
            "ok": True,
            "file_name": safe_name,
            "asr_text": asr_text,
            "translated_text": translated,
            "elapsed": {
                "upload_plus_total_ms_server": round((ts_done - ts_req_start) * 1000.0, 2),
                "model_only_ms_server": round((ts_done - ts_model_start) * 1000.0, 2),
            },
        }

    @app.websocket("/ws/ptt")
    async def ws_ptt(ws: WebSocket) -> None:
        await ws.accept()
        state = SessionState(pcm=bytearray(), sample_rate=args.sample_rate, ptt_started=False, ptt_up_server_ts=None)

        try:
            while True:
                try:
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

                            if len(state.pcm) < 3200:
                                state.ptt_started = False
                                await ws.send_text(json.dumps({"type": "error", "message": "No audio captured. Check mic permission."}))
                                continue

                            state.ptt_up_server_ts = time.perf_counter()
                            t_asr_start = time.perf_counter()
                            asr_text = await asr_backend.transcribe_pcm(bytes(state.pcm), state.sample_rate)
                            t_asr_done = time.perf_counter()

                            t_mt_start = time.perf_counter()
                            translated = await translator.translate(asr_text, args.target_lang, args.gemma_max_tokens)
                            t_mt_done = time.perf_counter()

                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "final",
                                        "asr_text": asr_text,
                                        "translated_text": translated,
                                        "elapsed": {
                                            "asr_ms": round((t_asr_done - t_asr_start) * 1000.0, 2),
                                            "mt_ms": round((t_mt_done - t_mt_start) * 1000.0, 2),
                                            "ptt_up_to_translated_ms_server": round((t_mt_done - state.ptt_up_server_ts) * 1000.0, 2),
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                            )

                            state.ptt_started = False

                        elif msg_type == "ping":
                            await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

                    elif "bytes" in msg and msg["bytes"] is not None:
                        if state.ptt_started:
                            state.pcm.extend(msg["bytes"])

                except Exception as exc:  # noqa: BLE001
                    state.ptt_started = False
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))

        except WebSocketDisconnect:
            return

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime PTT benchmark: whisper-small-komix -> gemma4 translation")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--sample-rate", default=16000, type=int)

    parser.add_argument("--whisper-model", default="seastar105/whisper-small-komixv2", type=str)
    parser.add_argument("--gemma-model", default="google/gemma-4-E2B-it", type=str)
    parser.add_argument("--gemma-max-tokens", default=128, type=int)
    parser.add_argument("--target-lang", default="Japanese", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()