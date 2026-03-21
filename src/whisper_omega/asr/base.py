from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from whisper_omega.runtime.models import BackendError, Segment, Word


@dataclass(slots=True)
class BackendTranscription:
    text: str
    language: str
    segments: list[Segment]
    words: list[Word]
    backend_errors: list[BackendError] = field(default_factory=list)


class ASRBackend:
    name = "base"

    def transcribe(
        self,
        audio_path: Path,
        model_name: str,
        language: str | None,
        device: str,
        batch_size: int | None = None,
    ) -> BackendTranscription:
        raise NotImplementedError

