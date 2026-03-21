#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NON_LATIN_RE = re.compile(r"[^\u0000-\u024f]")
TOKEN_RE = re.compile(r"[^\s、。！？,.!?;:]+")


def extract_non_latin_tokens(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("manifest must be a JSON array")
    tokens: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        transcription = row.get("transcription")
        if not isinstance(transcription, str):
            continue
        for token in TOKEN_RE.findall(transcription):
            if NON_LATIN_RE.search(token):
                tokens.add(token)
    return sorted(tokens)


def build_stub_map(manifest_path: Path) -> dict[str, str]:
    return {token: "" for token in extract_non_latin_tokens(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a generic alignment text-map template from a fixture manifest.json.")
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    payload = build_stub_map(args.manifest_path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
