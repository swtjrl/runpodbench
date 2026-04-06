#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import wave
from typing import Iterable


def make_sine_wave(path: pathlib.Path, duration_s: float, sample_rate: int = 16000) -> None:
    amplitude = 0.15
    frequency_hz = 220.0
    num_samples = int(duration_s * sample_rate)

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            value = amplitude * math.sin(2.0 * math.pi * frequency_hz * (i / sample_rate))
            sample = int(max(-1.0, min(1.0, value)) * 32767)
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))

        wav_file.writeframes(bytes(frames))


def parse_durations(raw: str) -> Iterable[float]:
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        yield float(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic WAV files for Gemma4 audio benchmarks")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("audio_samples"))
    parser.add_argument("--durations", type=str, default="1,3,5,8,12,15,20")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    durations = list(parse_durations(args.durations))

    for duration in durations:
        fname = f"tone_{str(duration).replace('.', 'p')}s.wav"
        out_path = args.out_dir / fname
        make_sine_wave(out_path, duration)
        print(f"generated: {out_path} ({duration}s)")


if __name__ == "__main__":
    main()