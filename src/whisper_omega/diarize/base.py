from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Iterable

from whisper_omega.runtime.models import BackendError, Segment, Speaker, Word


@dataclass(slots=True)
class DiarizationOutcome:
    segments: list[Segment]
    words: list[Word]
    speakers: list[Speaker]
    backend_errors: list[BackendError] = field(default_factory=list)


class DiarizationBackend:
    name = "base"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        raise NotImplementedError


class NoopDiarizationBackend(DiarizationBackend):
    name = "none"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        return DiarizationOutcome(segments=segments, words=words, speakers=[])


class UnavailablePyannoteBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        try:
            from pyannote.audio import Pipeline
        except Exception:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="DIARIZATION_BACKEND_UNAVAILABLE",
                        category="dependency",
                        message="pyannote.audio is not installed",
                        retryable=False,
                    )
                ],
            )
        if not os.environ.get("HF_TOKEN"):
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="HF_TOKEN_MISSING",
                        category="configuration",
                        message="HF_TOKEN is required for pyannote diarization",
                        retryable=False,
                    )
                ],
            )
        audio_path = os.environ.get("OMEGA_AUDIO_PATH")
        if not audio_path:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="CONFIG_INVALID",
                        category="configuration",
                        message="OMEGA_AUDIO_PATH is required for pyannote diarization",
                        retryable=False,
                    )
                ],
            )

        model_id = os.environ.get("OMEGA_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        device = os.environ.get("OMEGA_DEVICE", "cpu")
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
            if hasattr(pipeline, "to"):
                import torch

                torch_device = torch.device("cuda" if device == "cuda" else "cpu")
                pipeline.to(torch_device)
            diarization = pipeline(audio_path)
        except Exception as exc:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="DIARIZATION_BACKEND_UNAVAILABLE",
                        category="backend",
                        message=str(exc),
                        retryable=False,
                    )
                ],
            )

        speaker_turns = list(_iter_speaker_turns(diarization))
        if not speaker_turns:
            return DiarizationOutcome(segments=segments, words=words, speakers=[])

        assigned_segments = [_with_speaker(segment, _speaker_for_interval(segment.start, segment.end, speaker_turns)) for segment in segments]
        assigned_words = [_with_word_speaker(word, _speaker_for_interval(word.start, word.end, speaker_turns)) for word in words]
        speakers = [
            Speaker(id=speaker, start=start, end=end, label=speaker)
            for start, end, speaker in speaker_turns
        ]
        return DiarizationOutcome(
            segments=assigned_segments,
            words=assigned_words,
            speakers=speakers,
        )


def _iter_speaker_turns(diarization) -> Iterable[tuple[float, float, str]]:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        yield (float(turn.start), float(turn.end), str(speaker))


def _speaker_for_interval(start: float, end: float, speaker_turns: list[tuple[float, float, str]]) -> str | None:
    midpoint = (start + end) / 2
    best_overlap = 0.0
    best_speaker: str | None = None
    for turn_start, turn_end, speaker in speaker_turns:
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
        elif best_speaker is None and turn_start <= midpoint <= turn_end:
            best_speaker = speaker
    return best_speaker


def _with_speaker(segment: Segment, speaker: str | None) -> Segment:
    return Segment(
        id=segment.id,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        speaker=speaker,
    )


def _with_word_speaker(word: Word, speaker: str | None) -> Word:
    return Word(
        text=word.text,
        start=word.start,
        end=word.end,
        speaker=speaker,
        confidence=word.confidence,
    )
