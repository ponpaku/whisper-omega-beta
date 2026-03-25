from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from whisper_omega.runtime.models import TranscriptionResult
from whisper_omega.runtime.policy import PolicyConfig, resolve_timestamp_strategy
from whisper_omega.runtime.service import ServiceConfig, TranscriptionService


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    timestamp_strategy: str | None = None,
    word_timestamps: bool = True,
    include_segments: bool = True,
    include_words: bool = True,
) -> TranscriptionResult:
    normalized_required = list(required_features or [])
    resolved_timestamp_strategy = resolve_timestamp_strategy(
        timestamp_strategy,
        word_timestamps=word_timestamps,
        align_backend=align_backend,
        required_features=normalized_required,
    )
    config = ServiceConfig(
        policy=PolicyConfig(runtime_policy=runtime_policy, device=device),
        model_name=model_name,
        language=language,
        required_features=normalized_required,
        align_backend=align_backend,
        diarize_backend=diarize_backend,
        word_timestamps=word_timestamps,
        timestamp_strategy=resolved_timestamp_strategy,
        include_segments=include_segments,
        include_words=include_words,
    )
    service = TranscriptionService(config)
    diarization_env = {
        "OMEGA_PYANNOTE_NUM_SPEAKERS": None if num_speakers is None else str(num_speakers),
        "OMEGA_PYANNOTE_MIN_SPEAKERS": None if min_speakers is None else str(min_speakers),
        "OMEGA_PYANNOTE_MAX_SPEAKERS": None if max_speakers is None else str(max_speakers),
        "OMEGA_NEMO_NUM_SPEAKERS": None if num_speakers is None else str(num_speakers),
        "OMEGA_NEMO_MAX_SPEAKERS": None if max_speakers is None else str(max_speakers),
    }
    with _temporary_env(diarization_env):
        return service.transcribe(Path(audio_path))
