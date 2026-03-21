#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch

from whisper_omega.align.base import Wav2Vec2AlignmentBackend
from whisper_omega.runtime.models import Segment, Word


ROOT = Path(__file__).resolve().parents[1]
D1_MANIFEST = ROOT / "fixtures" / "d1_short_ja" / "manifest.json"
D2_MANIFEST = ROOT / "fixtures" / "d2_short_en" / "manifest.json"

_D1_JA_READING_MAP = {
    "1634": "にほんにはやくななせんのしまじまがありそのさいだいはほんしゅうれっとうでせかいでななばんめにおおきいしまとされています"
}


class Span:
    def __init__(self, start: int, end: int, score: float) -> None:
        self.start = start
        self.end = end
        self.score = score


class FakeModel:
    def __call__(self, waveform):
        _ = waveform
        return torch.zeros((1, 20, 4)), None


class FakeBundle:
    sample_rate = 16000

    def get_model(self):
        return FakeModel()

    def get_tokenizer(self):
        return lambda words: [[1] * max(1, len(word)) for word in words]

    def get_aligner(self):
        return lambda emission, tokens: [
            [Span(index * 5, (index + 1) * 5, 0.8 + (index * 0.05))] for index, _ in enumerate(tokens)
        ]


def _fake_modules() -> tuple[object, object]:
    fake_torchaudio = types.SimpleNamespace(
        pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
        functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
        load=lambda path: (torch.ones((1, 16000)), 16000),
    )
    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    return fake_torchaudio, fake_torch


def _load_manifest_entry(path: Path, index: int = 0) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[index]


def _build_words(text: str) -> list[Word]:
    return [Word(text=token, start=index * 0.1, end=(index + 1) * 0.1, speaker=None) for index, token in enumerate(text.split())]


def run_d2_short_en_smoke() -> dict:
    entry = _load_manifest_entry(D2_MANIFEST)
    audio_path = D2_MANIFEST.parent / entry["file"]
    backend = Wav2Vec2AlignmentBackend()
    fake_torchaudio, fake_torch = _fake_modules()

    with patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
        outcome = backend.align(
            audio_path=audio_path,
            text=entry["transcription"],
            segments=[Segment(id=0, start=0.0, end=1.0, text=entry["transcription"], speaker=None)],
            words=_build_words(entry["transcription"]),
            language="en",
        )

    return {
        "dataset": "D2_SHORT_EN",
        "fixture": entry["file"],
        "success": not outcome.backend_errors,
        "strategy": outcome.strategy,
        "token_source": outcome.token_source,
        "word_count": len(outcome.words),
        "error_codes": [error.code for error in outcome.backend_errors],
    }


def run_d1_short_ja_reading_map_smoke() -> dict:
    entry = _load_manifest_entry(D1_MANIFEST)
    audio_path = D1_MANIFEST.parent / entry["file"]
    backend = Wav2Vec2AlignmentBackend()
    fake_torchaudio, fake_torch = _fake_modules()
    reading = _D1_JA_READING_MAP.get(entry["id"], "")

    with tempfile.TemporaryDirectory() as tmpdir:
        mapping_path = Path(tmpdir) / "ja_reading_map.json"
        mapping_path.write_text(json.dumps({entry["transcription"]: reading}, ensure_ascii=False), encoding="utf-8")
        with patch.dict(
            "os.environ",
            {"OMEGA_ALIGNMENT_JA_READING_MAP": str(mapping_path)},
            clear=False,
        ), patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
            outcome = backend.align(
                audio_path=audio_path,
                text=entry["transcription"],
                segments=[Segment(id=0, start=0.0, end=1.0, text=entry["transcription"], speaker=None)],
                words=[Word(text=entry["transcription"], start=0.0, end=1.0, speaker=None)],
                language="ja",
            )

    return {
        "dataset": "D1_SHORT_JA",
        "fixture": entry["file"],
        "success": not outcome.backend_errors,
        "strategy": outcome.strategy,
        "token_source": outcome.token_source,
        "word_count": len(outcome.words),
        "error_codes": [error.code for error in outcome.backend_errors],
    }


def build_alignment_smoke_report() -> dict:
    checks = [run_d2_short_en_smoke(), run_d1_short_ja_reading_map_smoke()]
    return {
        "report_version": "0.1",
        "checks": checks,
        "all_passed": all(item["success"] for item in checks),
    }


def main() -> int:
    print(json.dumps(build_alignment_smoke_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
