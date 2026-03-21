#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import tempfile
import wave
from pathlib import PurePosixPath
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DATASET_CONFIGS = {
    "D1_SHORT_JA": {"dataset": "google/fleurs", "config": "ja_jp", "split": "validation", "repo_split": "dev"},
    "D2_SHORT_EN": {"dataset": "google/fleurs", "config": "en_us", "split": "validation", "repo_split": "dev"},
}
DATASET_REPO_URL = "https://huggingface.co/datasets/{dataset}/resolve/main/data/{config}/{name}"


def dataset_config(dataset_id: str) -> dict[str, str]:
    if dataset_id not in DATASET_CONFIGS:
        raise KeyError(f"unsupported dataset_id: {dataset_id}")
    return DATASET_CONFIGS[dataset_id]


def _to_mono_samples(array_like) -> list[float]:
    if hasattr(array_like, "tolist"):
        array_like = array_like.tolist()
    if not array_like:
        return []
    first = array_like[0]
    if isinstance(first, (list, tuple)):
        return [sum(frame) / len(frame) for frame in array_like]
    return list(array_like)


def _float_to_pcm16(samples: list[float]) -> bytes:
    payload = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm = int(clipped * 32767.0)
        payload.extend(int(pcm).to_bytes(2, "little", signed=True))
    return bytes(payload)


def repo_asset_url(cfg: dict[str, str], name: str) -> str:
    return DATASET_REPO_URL.format(dataset=cfg["dataset"], config=cfg["config"], name=name)


def parse_repo_tsv(tsv_text: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    reader = csv.reader(tsv_text.splitlines(), delimiter="\t")
    for row in reader:
        if len(row) < 3:
            continue
        rows.append(
            {
                "id": row[0],
                "file": row[1],
                "transcription": row[2],
            }
        )
        if len(rows) >= count:
            break
    return rows


def load_repo_rows(cfg: dict[str, str], count: int) -> list[dict[str, str]]:
    split_name = cfg["repo_split"]
    with urlopen(repo_asset_url(cfg, f"{split_name}.tsv")) as response:
        return parse_repo_tsv(response.read().decode("utf-8"), count)


def extract_repo_archive(cfg: dict[str, str], rows: list[dict[str, str]], output_dir: Path) -> list[dict]:
    split_name = cfg["repo_split"]
    wanted = {row["file"]: row for row in rows}
    manifest: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / f"{cfg['config']}_{split_name}.tar.gz"
        with urlopen(repo_asset_url(cfg, f"audio/{split_name}.tar.gz")) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                file_name = PurePosixPath(member.name).name
                row = wanted.get(file_name)
                if row is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                path = output_dir / file_name
                with path.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                manifest.append(
                    {
                        "file": file_name,
                        "sampling_rate": 16000,
                        "transcription": row["transcription"],
                        "id": row["id"],
                    }
                )
                if len(manifest) >= len(rows):
                    break
    if len(manifest) != len(rows):
        missing = sorted(set(wanted) - {entry["file"] for entry in manifest})
        raise RuntimeError(f"missing files in archive: {', '.join(missing)}")
    return manifest


def export_fixture_from_repo(cfg: dict[str, str], output_dir: Path, count: int) -> list[dict]:
    rows = load_repo_rows(cfg, count)
    return extract_repo_archive(cfg, rows, output_dir)


def export_fixture(dataset_id: str, output_dir: Path, count: int) -> list[dict]:
    cfg = dataset_config(dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception:
        return export_fixture_from_repo(cfg, output_dir, count)

    try:
        ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    except (ValueError, HTTPError, URLError, ConnectionError, OSError):
        return export_fixture_from_repo(cfg, output_dir, count)

    manifest: list[dict] = []
    for index, row in enumerate(ds.select(range(min(count, len(ds))))):
        audio = row["audio"]
        samples = _to_mono_samples(audio["array"])
        sample_rate = int(audio["sampling_rate"])
        file_name = f"{dataset_id.lower()}_{index:02d}.wav"
        path = output_dir / file_name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(_float_to_pcm16(samples))
        manifest.append(
            {
                "file": file_name,
                "sampling_rate": sample_rate,
                "transcription": row.get("transcription", ""),
                "id": row.get("id", index),
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export local validation fixtures from Google FLEURS.")
    parser.add_argument("dataset_id", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    manifest = export_fixture(args.dataset_id, args.output_dir, args.count)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_id": args.dataset_id, "count": len(manifest), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
