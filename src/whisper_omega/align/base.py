from __future__ import annotations

from dataclasses import dataclass, field

from whisper_omega.runtime.models import BackendError, Word


@dataclass(slots=True)
class AlignmentOutcome:
    words: list[Word]
    backend_errors: list[BackendError] = field(default_factory=list)


class AlignmentBackend:
    name = "base"

    def align(self, words: list[Word], language: str | None) -> AlignmentOutcome:
        raise NotImplementedError


class NoopAlignmentBackend(AlignmentBackend):
    name = "none"

    def align(self, words: list[Word], language: str | None) -> AlignmentOutcome:
        _ = language
        return AlignmentOutcome(words=words)


class UnavailableWav2Vec2Backend(AlignmentBackend):
    name = "wav2vec2"

    def align(self, words: list[Word], language: str | None) -> AlignmentOutcome:
        _ = language
        if words:
            return AlignmentOutcome(words=words)
        return AlignmentOutcome(
            words=words,
            backend_errors=[
                BackendError(
                    backend=self.name,
                    code="ALIGNMENT_MODEL_UNAVAILABLE",
                    category="backend",
                    message="alignment backend could not produce word timings",
                    retryable=False,
                )
            ],
        )
