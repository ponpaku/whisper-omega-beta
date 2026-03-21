#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path


def read_wav_mono(path: Path) -> tuple[int, list[float]]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"unsupported wav header: {path}")
        fmt_payload = None
        data_payload = None
        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:8], "little")
            payload = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                fmt_payload = payload
            elif chunk_id == b"data":
                data_payload = payload
        if fmt_payload is None or data_payload is None or len(fmt_payload) < 16:
            raise RuntimeError(f"incomplete wav chunks: {path}")
        audio_format = int.from_bytes(fmt_payload[0:2], "little")
        channels = int.from_bytes(fmt_payload[2:4], "little")
        sample_rate = int.from_bytes(fmt_payload[4:8], "little")
        bits_per_sample = int.from_bytes(fmt_payload[14:16], "little")
        if channels != 1:
            raise RuntimeError(f"only mono wav is supported: {path}")
        if audio_format == 1 and bits_per_sample == 16:
            return sample_rate, [value / 32768.0 for (value,) in struct.iter_unpack("<h", data_payload)]
        if audio_format == 3 and bits_per_sample == 32:
            return sample_rate, [float(value) for (value,) in struct.iter_unpack("<f", data_payload)]
        raise RuntimeError(f"unsupported wav encoding format={audio_format} bits={bits_per_sample}: {path}")


def write_pcm16_wav(path: Path, sample_rate: int, samples: list[float]) -> None:
    payload = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        payload.extend(int(clipped * 32767.0).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(payload))

def concatenate_wavs(inputs: list[Path], output_path: Path, gap_ms: int) -> dict:
    if not inputs:
        raise RuntimeError("at least one input wav is required")

    combined: list[float] = []
    sample_rate: int | None = None
    timeline: list[dict] = []
    current_sample = 0

    for path in inputs:
        path_rate, samples = read_wav_mono(path)
        sample_count = len(samples)
        if sample_rate is None:
            sample_rate = path_rate
        elif sample_rate != path_rate:
            raise RuntimeError("all wav inputs must share sample rate")
        timeline.append(
            {
                "file": path.name,
                "start": round(current_sample / sample_rate, 3),
                "end": round((current_sample + sample_count) / sample_rate, 3),
            }
        )
        combined.extend(samples)
        current_sample += sample_count
        if gap_ms > 0:
            gap_frames = int(sample_rate * (gap_ms / 1000.0))
            combined.extend([0.0] * gap_frames)
            current_sample += gap_frames

    if gap_ms > 0 and len(inputs) > 1:
        trim = int(sample_rate * (gap_ms / 1000.0))
        combined = combined[:-trim]
        current_sample -= trim

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(output_path, sample_rate, combined)

    recipe = {
        "output_file": output_path.name,
        "sample_rate": sample_rate,
        "channels": 1,
        "gap_ms": gap_ms,
        "duration": round(current_sample / sample_rate, 3),
        "inputs": timeline,
    }
    (output_path.parent / f"{output_path.stem}.recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate wav fixtures into a long-form validation audio file.")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--gap-ms", type=int, default=500)
    args = parser.parse_args()

    recipe = concatenate_wavs(args.inputs, args.output_path, args.gap_ms)
    print(json.dumps(recipe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
