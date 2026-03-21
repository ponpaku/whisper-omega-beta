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
            frame_count = len(data_payload) // 2
            samples = [value / 32768.0 for (value,) in struct.iter_unpack("<h", data_payload[: frame_count * 2])]
            return sample_rate, samples
        if audio_format == 3 and bits_per_sample == 32:
            frame_count = len(data_payload) // 4
            samples = [float(value) for (value,) in struct.iter_unpack("<f", data_payload[: frame_count * 4])]
            return sample_rate, samples
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


def build_mixture(
    speaker_a: Path,
    speaker_b: Path,
    output_path: Path,
    offset_ms: int,
    gain_a: float,
    gain_b: float,
) -> dict:
    sample_rate_a, samples_a = read_wav_mono(speaker_a)
    sample_rate_b, samples_b = read_wav_mono(speaker_b)
    if sample_rate_a != sample_rate_b:
        raise RuntimeError("sample rates must match for diarization fixture mixing")

    offset_frames = max(0, int(sample_rate_a * (offset_ms / 1000.0)))
    total_frames = max(len(samples_a), offset_frames + len(samples_b))
    mixed = [0.0] * total_frames

    for index, sample in enumerate(samples_a):
        mixed[index] += sample * gain_a
    for index, sample in enumerate(samples_b):
        mixed[index + offset_frames] += sample * gain_b

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(output_path, sample_rate_a, mixed)

    recipe = {
        "output_file": output_path.name,
        "sample_rate": sample_rate_a,
        "offset_ms": offset_ms,
        "speakers": [
            {"id": "SPEAKER_A", "source": speaker_a.name, "start": 0.0, "end": round(len(samples_a) / sample_rate_a, 3)},
            {
                "id": "SPEAKER_B",
                "source": speaker_b.name,
                "start": round(offset_frames / sample_rate_a, 3),
                "end": round((offset_frames + len(samples_b)) / sample_rate_a, 3),
            },
        ],
    }
    (output_path.parent / f"{output_path.stem}.recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic diarization validation fixture from two mono wav files.")
    parser.add_argument("speaker_a", type=Path)
    parser.add_argument("speaker_b", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--offset-ms", type=int, default=1200)
    parser.add_argument("--gain-a", type=float, default=0.75)
    parser.add_argument("--gain-b", type=float, default=0.75)
    args = parser.parse_args()

    recipe = build_mixture(
        speaker_a=args.speaker_a,
        speaker_b=args.speaker_b,
        output_path=args.output_path,
        offset_ms=args.offset_ms,
        gain_a=args.gain_a,
        gain_b=args.gain_b,
    )
    print(json.dumps(recipe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
