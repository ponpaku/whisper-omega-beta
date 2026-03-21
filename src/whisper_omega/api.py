from __future__ import annotations

from pathlib import Path

from whisper_omega.runtime.models import TranscriptionResult
from whisper_omega.runtime.policy import PolicyConfig
from whisper_omega.runtime.service import ServiceConfig, TranscriptionService


def transcribe_file(
    audio_path: str | Path,
    *,
    runtime_policy: str = "permissive",
    device: str = "auto",
    model_name: str = "small",
    language: str | None = None,
    required_features: list[str] | None = None,
    align_backend: str = "none",
    diarize_backend: str = "none",
) -> TranscriptionResult:
    config = ServiceConfig(
        policy=PolicyConfig(runtime_policy=runtime_policy, device=device),
        model_name=model_name,
        language=language,
        required_features=list(required_features or []),
        align_backend=align_backend,
        diarize_backend=diarize_backend,
    )
    service = TranscriptionService(config)
    return service.transcribe(Path(audio_path))
