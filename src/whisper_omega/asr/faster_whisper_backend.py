from __future__ import annotations

from pathlib import Path

from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.runtime.models import BackendError, Segment, Word


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def transcribe(
        self,
        audio_path: Path,
        model_name: str,
        language: str | None,
        device: str,
        batch_size: int | None = None,
        word_timestamps: bool = True,
    ) -> BackendTranscription:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("DEPENDENCY_MISSING:faster-whisper") from exc

        model = WhisperModel(model_name, device=device, compute_type="default")
        transcribe_kwargs = {
            "language": language,
            "word_timestamps": word_timestamps,
        }
        _ = batch_size
        segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)

        segments: list[Segment] = []
        words: list[Word] = []
        text_parts: list[str] = []
        for index, raw_segment in enumerate(segments_iter):
            seg_text = raw_segment.text.strip()
            text_parts.append(seg_text)
            segments.append(
                Segment(
                    id=index,
                    start=float(raw_segment.start),
                    end=float(raw_segment.end),
                    text=seg_text,
                    speaker=None,
                )
            )
            for raw_word in raw_segment.words or []:
                words.append(
                    Word(
                        text=raw_word.word.strip(),
                        start=float(raw_word.start),
                        end=float(raw_word.end),
                        speaker=None,
                        confidence=getattr(raw_word, "probability", None),
                    )
                )

        return BackendTranscription(
            text=" ".join(part for part in text_parts if part).strip(),
            language=(language or getattr(info, "language", "") or "").strip(),
            segments=segments,
            words=words,
            backend_errors=[],
        )


def dependency_error(backend: str, message: str) -> BackendError:
    return BackendError(
        backend=backend,
        code="DEPENDENCY_MISSING",
        category="dependency",
        message=message,
        retryable=False,
    )
