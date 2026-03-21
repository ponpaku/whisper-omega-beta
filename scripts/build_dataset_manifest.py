#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import struct
import wave
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(path: Path) -> str:
    if path.suffix.lower() != ".wav":
        return ""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                return ""
            return f"{frames / rate:.3f}"
    except Exception:
        return _riff_duration_seconds(path)


def _riff_duration_seconds(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return ""
            sample_rate = 0
            block_align = 0
            data_size = 0
            while True:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id = chunk_header[:4]
                chunk_size = int.from_bytes(chunk_header[4:8], "little")
                payload = handle.read(chunk_size)
                if len(payload) < chunk_size:
                    return ""
                if chunk_id == b"fmt " and len(payload) >= 16:
                    sample_rate = struct.unpack_from("<I", payload, 4)[0]
                    block_align = struct.unpack_from("<H", payload, 12)[0]
                elif chunk_id == b"data":
                    data_size = chunk_size
                if chunk_size % 2 == 1:
                    handle.read(1)
            if sample_rate <= 0 or block_align <= 0 or data_size <= 0:
                return ""
            frames = data_size / block_align
            return f"{frames / sample_rate:.3f}"
    except Exception:
        return ""


def collect_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def build_markdown(root: Path, dataset_id: str) -> str:
    lines = [
        "| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in collect_files(root):
        rel = path.relative_to(root)
        lines.append(
            f"| {dataset_id} | {rel.as_posix()} | {sha256sum(path)} | {duration_seconds(path)} | | | |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a markdown dataset manifest table.")
    parser.add_argument("root", type=Path, help="Dataset directory")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID such as D1_SHORT_JA")
    args = parser.parse_args()

    print(build_markdown(args.root.resolve(), args.dataset_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
