#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import statistics
import time
from dataclasses import dataclass

import torch
from openai import OpenAI
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


@dataclass(frozen=True)
class BenchRow:
    file_name: str
    asr_ms: float
    translation_ms: float
    total_ms: float


def load_asr_pipeline(model_id: str):
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

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


def run_once(asr_pipe, audio_path: pathlib.Path, client: OpenAI, gemma_model: str, max_tokens: int) -> BenchRow:
    t0 = time.perf_counter()
    asr_out = asr_pipe(str(audio_path))
    asr_text = asr_out["text"].strip()
    t1 = time.perf_counter()

    prompt = (
        "Translate the following Korean transcription to natural English. "
        "Return only English translation.\n"
        f"\nKorean:\n{asr_text}"
    )

    _ = client.chat.completions.create(
        model=gemma_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    t2 = time.perf_counter()

    return BenchRow(
        file_name=audio_path.name,
        asr_ms=(t1 - t0) * 1000.0,
        translation_ms=(t2 - t1) * 1000.0,
        total_ms=(t2 - t0) * 1000.0,
    )


def print_summary(rows: list[BenchRow]) -> None:
    asr_vals = [r.asr_ms for r in rows]
    tr_vals = [r.translation_ms for r in rows]
    total_vals = [r.total_ms for r in rows]

    print("\n=== whisper-small-komixv2 -> gemma4-e2b benchmark ===")
    print(f"samples: {len(rows)}")
    print(f"ASR mean_ms: {statistics.mean(asr_vals):.2f} / p50_ms: {statistics.median(asr_vals):.2f}")
    print(f"Translate mean_ms: {statistics.mean(tr_vals):.2f} / p50_ms: {statistics.median(tr_vals):.2f}")
    print(f"Total mean_ms: {statistics.mean(total_vals):.2f} / p50_ms: {statistics.median(total_vals):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Whisper small komix + Gemma4 translation latency")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=pathlib.Path("audio_samples"))
    parser.add_argument("--audio-glob", type=str, default="*.wav")
    parser.add_argument("--whisper-model", type=str, default="seastar105/whisper-small-komixv2")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--gemma-model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    audio_files = sorted(args.audio_dir.glob(args.audio_glob))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found at {args.audio_dir} with glob {args.audio_glob}")

    print(f"Loading ASR model: {args.whisper_model}")
    load_t0 = time.perf_counter()
    asr_pipe = load_asr_pipeline(args.whisper_model)
    load_t1 = time.perf_counter()
    print(f"ASR load time: {(load_t1 - load_t0):.2f}s")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows: list[BenchRow] = []
    for _ in range(args.rounds):
        for audio_path in audio_files:
            row = run_once(asr_pipe, audio_path, client, args.gemma_model, args.max_tokens)
            rows.append(row)
            print(
                f"{row.file_name} asr_ms={row.asr_ms:.2f} "
                f"translate_ms={row.translation_ms:.2f} total_ms={row.total_ms:.2f}"
            )

    print_summary(rows)


if __name__ == "__main__":
    main()