#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Sequence

from openai import AsyncOpenAI


@dataclass(frozen=True)
class RequestResult:
    file_name: str
    duration_s: float
    elapsed_ms: float
    ok: bool
    error: str


def parse_duration_from_name(path: pathlib.Path) -> float:
    # Expect names like tone_12s.wav or tone_1p5s.wav
    stem = path.stem
    if "_" not in stem:
        return -1.0
    value = stem.split("_")[-1].replace("s", "").replace("p", ".")
    try:
        return float(value)
    except ValueError:
        return -1.0


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def start_local_http_server(directory: pathlib.Path, port: int) -> ThreadingHTTPServer:
    handler_cls = lambda *args, **kwargs: QuietHandler(*args, directory=str(directory), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def one_request(
    client: AsyncOpenAI,
    model: str,
    audio_url: str,
    file_name: str,
    duration_s: float,
    max_tokens: int,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {
                            "type": "text",
                            "text": "Transcribe this audio verbatim. If not speech, briefly describe the sound.",
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(file_name=file_name, duration_s=duration_s, elapsed_ms=elapsed_ms, ok=True, error="")
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(file_name=file_name, duration_s=duration_s, elapsed_ms=elapsed_ms, ok=False, error=str(exc))


async def run_batch(
    client: AsyncOpenAI,
    model: str,
    audio_files: Sequence[pathlib.Path],
    server_port: int,
    requests_per_round: int,
    rounds: int,
    concurrency: int,
    max_tokens: int,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def bounded(file_path: pathlib.Path) -> RequestResult:
        async with sem:
            audio_url = f"http://127.0.0.1:{server_port}/{file_path.name}"
            duration_s = parse_duration_from_name(file_path)
            return await one_request(
                client=client,
                model=model,
                audio_url=audio_url,
                file_name=file_path.name,
                duration_s=duration_s,
                max_tokens=max_tokens,
            )

    for _ in range(rounds):
        sampled = [random.choice(audio_files) for _ in range(requests_per_round)]
        round_results = await asyncio.gather(*(bounded(path) for path in sampled))
        results.extend(round_results)

    return results


def summarize(results: Sequence[RequestResult], title: str) -> None:
    ok_results = [r for r in results if r.ok]
    err_results = [r for r in results if not r.ok]

    print(f"\n=== {title} ===")
    print(f"total: {len(results)}")
    print(f"ok: {len(ok_results)}")
    print(f"errors: {len(err_results)}")

    if ok_results:
        elapsed = [r.elapsed_ms for r in ok_results]
        print(f"mean_ms: {statistics.mean(elapsed):.2f}")
        print(f"p50_ms: {statistics.median(elapsed):.2f}")
        if len(elapsed) >= 20:
            print(f"p95_ms: {statistics.quantiles(elapsed, n=20)[-1]:.2f}")
        print(f"max_ms: {max(elapsed):.2f}")


def write_csv(path: pathlib.Path, rows: Sequence[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file_name", "duration_s", "elapsed_ms", "ok", "error"])
        for row in rows:
            writer.writerow([row.file_name, row.duration_s, f"{row.elapsed_ms:.3f}", row.ok, row.error])


def choose_single_file(audio_files: Sequence[pathlib.Path], name: str | None) -> pathlib.Path:
    if name:
        for file_path in audio_files:
            if file_path.name == name:
                return file_path
        raise FileNotFoundError(f"single file not found: {name}")
    return sorted(audio_files, key=parse_duration_from_name)[0]


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Gemma4 audio benchmark: single-latency + mixed-length batch stress")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=pathlib.Path("audio_samples"))
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--single-rounds", type=int, default=5)
    parser.add_argument("--single-file", type=str, default=None)
    parser.add_argument("--batch-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("results/gemma4_audio_bench.csv"))
    args = parser.parse_args()

    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No wav files in {args.audio_dir}. Run make_audio_samples.py first.")

    http_server = start_local_http_server(args.audio_dir.resolve(), args.http_port)
    print(f"Serving local audio files: http://127.0.0.1:{args.http_port}/")

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    all_results: list[RequestResult] = []

    target = choose_single_file(audio_files, args.single_file)
    single_results: list[RequestResult] = []
    for _ in range(args.single_rounds):
        url = f"http://127.0.0.1:{args.http_port}/{target.name}"
        single_results.append(
            await one_request(
                client=client,
                model=args.model,
                audio_url=url,
                file_name=target.name,
                duration_s=parse_duration_from_name(target),
                max_tokens=args.max_tokens,
            )
        )
    all_results.extend(single_results)
    summarize(single_results, f"single-file latency ({target.name})")

    batch_results = await run_batch(
        client=client,
        model=args.model,
        audio_files=audio_files,
        server_port=args.http_port,
        requests_per_round=args.batch_size,
        rounds=args.batch_rounds,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    )
    all_results.extend(batch_results)
    summarize(batch_results, "mixed-length batch stress")

    write_csv(args.csv, all_results)
    print(f"\nCSV written: {args.csv}")

    http_server.shutdown()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()