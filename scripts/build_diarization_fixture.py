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


def build_multispeaker_mixture(
    tracks: list[dict],
    output_path: Path,
) -> dict:
    if not tracks:
        raise RuntimeError("at least one track is required")

    loaded_tracks = []
    sample_rate = None
    for track in tracks:
        source = Path(track["source"])
        track_sample_rate, samples = read_wav_mono(source)
        if sample_rate is None:
            sample_rate = track_sample_rate
        elif sample_rate != track_sample_rate:
            raise RuntimeError("sample rates must match for diarization fixture mixing")
        offset_frames = max(0, int(track_sample_rate * (int(track.get("offset_ms", 0)) / 1000.0)))
        loaded_tracks.append(
            {
                "id": track.get("id", source.stem),
                "source": source,
                "offset_frames": offset_frames,
                "gain": float(track.get("gain", 0.75)),
                "samples": samples,
            }
        )

    assert sample_rate is not None
    total_frames = max(track["offset_frames"] + len(track["samples"]) for track in loaded_tracks)
    mixed = [0.0] * total_frames

    for track in loaded_tracks:
        for index, sample in enumerate(track["samples"]):
            mixed[index + track["offset_frames"]] += sample * track["gain"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(output_path, sample_rate, mixed)

    recipe = {
        "output_file": output_path.name,
        "sample_rate": sample_rate,
        "tracks": [
            {
                "id": track["id"],
                "source": track["source"].name,
                "offset_ms": round((track["offset_frames"] / sample_rate) * 1000.0),
                "start": round(track["offset_frames"] / sample_rate, 3),
                "end": round((track["offset_frames"] + len(track["samples"])) / sample_rate, 3),
            }
            for track in loaded_tracks
        ],
    }
    (output_path.parent / f"{output_path.stem}.recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recipe


def build_mixture(
    speaker_a: Path,
    speaker_b: Path,
    output_path: Path,
    offset_ms: int,
    gain_a: float,
    gain_b: float,
) -> dict:
    return build_multispeaker_mixture(
        [
            {"id": "SPEAKER_A", "source": speaker_a, "offset_ms": 0, "gain": gain_a},
            {"id": "SPEAKER_B", "source": speaker_b, "offset_ms": offset_ms, "gain": gain_b},
        ],
        output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic diarization validation fixture from two mono wav files.")
    parser.add_argument("speaker_a", type=Path)
    parser.add_argument("speaker_b", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--offset-ms", type=int, default=1200)
    parser.add_argument("--gain-a", type=float, default=0.75)
    parser.add_argument("--gain-b", type=float, default=0.75)
    parser.add_argument("--speaker-c", type=Path)
    parser.add_argument("--offset-ms-c", type=int, default=2400)
    parser.add_argument("--gain-c", type=float, default=0.75)
    args = parser.parse_args()

    tracks = [
        {"id": "SPEAKER_A", "source": args.speaker_a, "offset_ms": 0, "gain": args.gain_a},
        {"id": "SPEAKER_B", "source": args.speaker_b, "offset_ms": args.offset_ms, "gain": args.gain_b},
    ]
    if args.speaker_c:
        tracks.append({"id": "SPEAKER_C", "source": args.speaker_c, "offset_ms": args.offset_ms_c, "gain": args.gain_c})

    recipe = build_multispeaker_mixture(tracks, args.output_path)
    print(json.dumps(recipe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
