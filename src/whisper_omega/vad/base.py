from __future__ import annotations

from dataclasses import dataclass, field

from whisper_omega.runtime.models import BackendError


@dataclass(slots=True)
class SpeechRegion:
    start: float
    end: float


@dataclass(slots=True)
class VADOutcome:
    regions: list[SpeechRegion]
    backend_errors: list[BackendError] = field(default_factory=list)


class VADBackend:
    name = "base"

    def detect(self, audio_path: str) -> VADOutcome:
        raise NotImplementedError


class NoopVADBackend(VADBackend):
    name = "none"

    def detect(self, audio_path: str) -> VADOutcome:
        _ = audio_path
        return VADOutcome(regions=[])

