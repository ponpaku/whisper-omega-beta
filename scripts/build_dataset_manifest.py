#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import wave
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DATASET_DEFAULTS = {
    "D1_SHORT_JA": {"language": "ja", "speakers": "1"},
    "D2_SHORT_EN": {"language": "en", "speakers": "1"},
    "D3_LONG_MIXED": {"language": "ja+en", "speakers": "1"},
    "D4_DIARIZATION": {"language": "mixed", "speakers": "2-4"},
    "D5_FAILURE_INJECTION": {"language": "n/a", "speakers": "n/a"},
}


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


def _load_manifest_entries(root: Path) -> dict[str, dict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    entries: dict[str, dict] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            entries[item["file"]] = item
    return entries


def _recipe_metadata(path: Path) -> dict:
    recipe_path = path.with_suffix(".recipe.json")
    if not recipe_path.exists():
        return {}
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _row_metadata(root: Path, path: Path, dataset_id: str, manifest_entries: dict[str, dict]) -> tuple[str, str, str]:
    defaults = DATASET_DEFAULTS.get(dataset_id, {})
    language = defaults.get("language", "")
    speakers = defaults.get("speakers", "")
    notes = ""

    manifest_entry = manifest_entries.get(path.name, {})
    if manifest_entry:
        notes = manifest_entry.get("note", "")
        expected_error = manifest_entry.get("expected_error_code")
        if expected_error:
            notes = f"{notes}; expected={expected_error}".strip("; ")

    recipe = _recipe_metadata(path)
    if recipe:
        if isinstance(recipe.get("speakers"), list):
            speakers = str(len(recipe["speakers"]))
        elif isinstance(recipe.get("inputs"), list):
            input_count = len(recipe["inputs"])
            if input_count > 0:
                notes = f"concatenated_inputs={input_count}" if not notes else f"{notes}; concatenated_inputs={input_count}"
        gap_ms = recipe.get("gap_ms")
        offset_ms = recipe.get("offset_ms")
        if gap_ms is not None:
            notes = f"gap_ms={gap_ms}" if not notes else f"{notes}; gap_ms={gap_ms}"
        if offset_ms is not None:
            notes = f"offset_ms={offset_ms}" if not notes else f"{notes}; offset_ms={offset_ms}"

    return language, speakers, notes


def build_markdown(root: Path, dataset_id: str) -> str:
    manifest_entries = _load_manifest_entries(root)
    lines = [
        "| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in collect_files(root):
        rel = path.relative_to(root)
        language, speakers, notes = _row_metadata(root, path, dataset_id, manifest_entries)
        lines.append(
            f"| {dataset_id} | {rel.as_posix()} | {sha256sum(path)} | {duration_seconds(path)} | {language} | {speakers} | {notes} |"
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
