from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResultStatus = Literal["success", "degraded", "failure"]
ErrorCategory = Literal["usage", "dependency", "configuration", "runtime", "backend", "validation", "internal"]
FallbackType = Literal["device", "backend", "feature", "quality"]
FeatureName = Literal["asr", "vad", "alignment", "diarization", "subtitle_export"]


def _rounded(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


@dataclass(slots=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": _rounded(self.start, 3),
            "end": _rounded(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str | None
    confidence: float | None = None

    def to_dict(self) -> dict:
        payload = {
            "text": self.text,
            "start": _rounded(self.start, 3),
            "end": _rounded(self.end, 3),
            "speaker": self.speaker,
            "confidence": _rounded(self.confidence, 4),
        }
        return payload


@dataclass(slots=True)
class BackendError:
    backend: str
    code: str
    category: ErrorCategory
    message: str
    retryable: bool

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(slots=True)
class Fallback:
    type: FallbackType
    from_value: str
    to_value: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "from": self.from_value,
            "to": self.to_value,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Metadata:
    asr_backend: str
    align_backend: str
    diarization_backend: str
    device: str
    requested_device: str
    actual_device: str
    fallbacks: list[Fallback] = field(default_factory=list)
    requested_features: list[str] = field(default_factory=list)
    completed_features: list[str] = field(default_factory=list)
    failed_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asr_backend": self.asr_backend,
            "align_backend": self.align_backend,
            "diarization_backend": self.diarization_backend,
            "device": self.device,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "fallbacks": [item.to_dict() for item in self.fallbacks],
            "requested_features": self.requested_features,
            "completed_features": self.completed_features,
            "failed_features": self.failed_features,
        }


@dataclass(slots=True)
class TranscriptionResult:
    schema_version: str
    status: ResultStatus
    text: str
    language: str
    segments: list[Segment]
    words: list[Word]
    speakers: list[str]
    metadata: Metadata
    error_code: str | None = None
    error_category: ErrorCategory | None = None
    backend_errors: list[BackendError] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("schema_version must be 1.0.0")
        for segment in self.segments:
            if segment.speaker == "":
                raise ValueError("segment speaker must be null or non-empty")
        for word in self.words:
            if word.speaker == "":
                raise ValueError("word speaker must be null or non-empty")
        if self.status == "success":
            if self.error_code is not None or self.error_category is not None or self.backend_errors:
                raise ValueError("success result cannot carry errors")
        if self.status == "failure":
            if not self.error_code or not self.error_category:
                raise ValueError("failure result requires error_code and error_category")

    @property
    def exit_family(self) -> str:
        if self.status == "success":
            return "success"
        if self.status == "degraded":
            return "degraded"
        if self.error_category == "dependency":
            return "dependency"
        if self.error_category == "configuration":
            return "configuration"
        if self.error_category == "usage":
            return "usage"
        return "runtime"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "text": self.text,
            "language": self.language,
            "segments": [item.to_dict() for item in self.segments],
            "words": [item.to_dict() for item in self.words],
            "speakers": self.speakers,
            "metadata": self.metadata.to_dict(),
            "error_code": self.error_code,
            "error_category": self.error_category,
            "backend_errors": [item.to_dict() for item in self.backend_errors],
        }

