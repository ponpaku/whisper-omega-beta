#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from unittest.mock import patch

import torch

from whisper_omega.diarize.base import UnavailablePyannoteBackend
from whisper_omega.runtime.models import Segment, Word


ROOT = Path(__file__).resolve().parents[1]
D4_FIXTURES = [
    ROOT / "fixtures" / "d4_diarization" / "d4_mix_01.recipe.json",
    ROOT / "fixtures" / "d4_diarization" / "d4_mix_3spk_01.recipe.json",
]


class FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeDiarization:
    def __init__(self, tracks: list[dict]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        _ = yield_label
        for track in self._tracks:
            yield (FakeTurn(track["start"], track["end"]), None, track["id"])


def _run_recipe(recipe_path: Path) -> dict:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    audio_path = recipe_path.with_suffix(".wav")
    tracks = recipe["tracks"]

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, token=None):
            _ = (model_id, token)
            return cls()

        def __call__(self, audio_input, **kwargs):
            _ = (audio_input, kwargs)
            return FakeDiarization(tracks)

    fake_torchaudio = types.SimpleNamespace(load=lambda path: (torch.ones((1, 16000)), 16000))
    segments = []
    words = []
    for index, track in enumerate(tracks):
        if index == 0:
            anchor = track["start"] + 0.4
        else:
            anchor = track["end"] - 0.4
        segment_start = max(track["start"] + 0.05, anchor - 0.2)
        segment_end = min(track["end"] - 0.05, anchor + 0.2)
        word_start = max(track["start"] + 0.1, anchor - 0.1)
        word_end = min(track["end"] - 0.1, anchor + 0.1)
        segments.append(Segment(id=index, start=segment_start, end=segment_end, text=track["id"], speaker=None))
        words.append(Word(text=track["id"], start=word_start, end=word_end, speaker=None, confidence=0.9))

    with patch.dict(
        "os.environ",
        {
            "HF_TOKEN": "smoke-token",
            "OMEGA_AUDIO_PATH": str(audio_path),
        },
        clear=False,
    ), patch.dict(
        "sys.modules",
        {
            "pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline),
            "torchaudio": fake_torchaudio,
        },
    ):
        outcome = UnavailablePyannoteBackend().diarize(segments, words)

    assigned_speakers = {segment.speaker for segment in outcome.segments if segment.speaker}
    return {
        "fixture": audio_path.name,
        "track_count": len(tracks),
        "success": not outcome.backend_errors,
        "speaker_ids": sorted(speaker.id for speaker in outcome.speakers),
        "assigned_speakers": sorted(assigned_speakers),
        "error_codes": [error.code for error in outcome.backend_errors],
    }


def build_diarization_smoke_report() -> dict:
    checks = [_run_recipe(path) for path in D4_FIXTURES]
    return {
        "report_version": "0.1",
        "checks": checks,
        "all_passed": all(item["success"] and len(item["speaker_ids"]) == item["track_count"] for item in checks),
    }


def main() -> int:
    print(json.dumps(build_diarization_smoke_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
