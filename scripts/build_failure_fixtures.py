#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_failure_fixtures(source_wav: Path, output_dir: Path) -> list[dict]:
    payload = source_wav.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = [
        {
            "file": "empty.wav",
            "expected_error_code": "AUDIO_DECODE_FAILURE",
            "note": "zero-byte wav placeholder",
            "bytes": b"",
        },
        {
            "file": "truncated.wav",
            "expected_error_code": "AUDIO_DECODE_FAILURE",
            "note": "truncated wav payload",
            "bytes": payload[: max(16, min(len(payload), 256))],
        },
        {
            "file": "not_audio.wav",
            "expected_error_code": "AUDIO_DECODE_FAILURE",
            "note": "plain-text file with wav extension",
            "bytes": b"this is not audio data\n",
        },
    ]

    manifest: list[dict] = []
    for fixture in fixtures:
        path = output_dir / fixture["file"]
        path.write_bytes(fixture["bytes"])
        manifest.append(
            {
                "file": fixture["file"],
                "expected_error_code": fixture["expected_error_code"],
                "note": fixture["note"],
            }
        )

    manifest.append(
        {
            "file": "missing.wav",
            "expected_error_code": "AUDIO_DECODE_FAILURE",
            "note": "intentionally absent path for missing-file validation",
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local failure-injection fixtures from a source wav.")
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    manifest = build_failure_fixtures(args.source_wav, args.output_dir)
    print(json.dumps({"count": len(manifest), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
