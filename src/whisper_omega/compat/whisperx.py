from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WhisperXCompatRequest:
    runtime_policy: str
    require_alignment: bool
    require_diarization: bool
    diarize_backend: str
    align_backend: str
    output_format: str
    batch_size: int | None


def map_whisperx_options(
    diarize: bool,
    align_model: str | None,
    output_format: str,
    batch_size: int | None,
) -> WhisperXCompatRequest:
    return WhisperXCompatRequest(
        runtime_policy="permissive",
        require_alignment=align_model is not None,
        require_diarization=diarize,
        diarize_backend="pyannote" if diarize else "none",
        align_backend="wav2vec2" if align_model else "none",
        output_format=output_format,
        batch_size=batch_size,
    )
