#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import random
import statistics
import time
from dataclasses import dataclass

import requests


@dataclass
class Row:
    file_name: str
    elapsed_ms_client: float
    ok: bool
    status_code: int
    error: str


def one_request(base_url: str, audio_path: pathlib.Path, timeout_s: float) -> Row:
    t0 = time.perf_counter()
    try:
        with audio_path.open("rb") as fp:
            resp = requests.post(
                f"{base_url}/api/e2b/audio_once",
                files={"audio": (audio_path.name, fp, "audio/wav")},
                timeout=timeout_s,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.ok:
            return Row(audio_path.name, elapsed_ms, True, resp.status_code, "")
        return Row(audio_path.name, elapsed_ms, False, resp.status_code, resp.text[:300])
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Row(audio_path.name, elapsed_ms, False, 0, str(exc))


def print_summary(rows: list[Row], title: str) -> None:
    ok_rows = [r for r in rows if r.ok]
    err_rows = [r for r in rows if not r.ok]
    print(f"\n=== {title} ===")
    print(f"total={len(rows)} ok={len(ok_rows)} error={len(err_rows)}")
    if ok_rows:
        vals = [r.elapsed_ms_client for r in ok_rows]
        print(f"mean_ms={statistics.mean(vals):.2f}")
        print(f"p50_ms={statistics.median(vals):.2f}")
        if len(vals) >= 20:
            print(f"p95_ms={statistics.quantiles(vals, n=20)[-1]:.2f}")
        print(f"max_ms={max(vals):.2f}")


def run_single(base_url: str, audio_file: pathlib.Path, rounds: int, timeout_s: float) -> list[Row]:
    rows = []
    for _ in range(rounds):
        rows.append(one_request(base_url, audio_file, timeout_s))
    return rows


def run_stress(
    base_url: str,
    audio_files: list[pathlib.Path],
    total_requests: int,
    concurrency: int,
    timeout_s: float,
) -> list[Row]:
    rows: list[Row] = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(one_request, base_url, random.choice(audio_files), timeout_s)
            for _ in range(total_requests)
        ]
        for fut in cf.as_completed(futs):
            rows.append(fut.result())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Client benchmark: local PC -> Runpod upload -> E2B result latency")
    parser.add_argument("--base-url", type=str, required=True, help="example: http://<RUNPOD_IP>:9000")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=pathlib.Path("audio_samples"))
    parser.add_argument("--single-file", type=str, default=None)
    parser.add_argument("--single-rounds", type=int, default=5)
    parser.add_argument("--stress-total-requests", type=int, default=64)
    parser.add_argument("--stress-concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No wav files in {args.audio_dir}")

    if args.single_file:
        target = next((p for p in audio_files if p.name == args.single_file), None)
        if target is None:
            raise FileNotFoundError(f"single file not found: {args.single_file}")
    else:
        target = audio_files[0]

    single_rows = run_single(args.base_url, target, args.single_rounds, args.timeout)
    print_summary(single_rows, f"Goal2 single non-realtime ({target.name})")

    stress_rows = run_stress(
        args.base_url,
        audio_files,
        args.stress_total_requests,
        args.stress_concurrency,
        args.timeout,
    )
    print_summary(stress_rows, f"Goal3 stress batch/concurrency={args.stress_concurrency}")


if __name__ == "__main__":
    main()