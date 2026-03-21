from __future__ import annotations

import unittest
from unittest.mock import patch

from whisper_omega.align.base import AlignmentOutcome, UnavailableWav2Vec2Backend
from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.diarize.base import DiarizationBackend, DiarizationOutcome
from whisper_omega.runtime.models import BackendError, Segment, Word
from whisper_omega.runtime.policy import PolicyConfig
from whisper_omega.runtime.service import ServiceConfig, TranscriptionService


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None):
        _ = (audio_path, model_name, language, device, batch_size)
        return BackendTranscription(
            text="hello world",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker=None)],
            words=[Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
        )


class MissingDiarizationBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments, words):
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


class ServiceTests(unittest.TestCase):
    def test_permissive_optional_failures_become_degraded(self) -> None:
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                required_features=["diarization"],
                diarize_backend="pyannote",
            ),
            asr_backend=StubBackend(),
            diarization_backend=MissingDiarizationBackend(),
        )
        result = service.transcribe(__file__)
        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.metadata.failed_features, ["diarization"])
        self.assertEqual(len(result.backend_errors), 1)

    def test_strict_alignment_succeeds_with_existing_words(self) -> None:
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="strict", device="cpu"),
                required_features=["alignment"],
                align_backend="wav2vec2",
            ),
            asr_backend=StubBackend(),
            alignment_backend=UnavailableWav2Vec2Backend(),
        )
        result = service.transcribe(__file__)
        self.assertEqual(result.status, "success")
        self.assertIn("alignment", result.metadata.completed_features)
        self.assertEqual(result.metadata.alignment_strategy, "fallback-existing-words")
        self.assertEqual(result.metadata.alignment_token_source, "asr_words")

    def test_strict_gpu_with_auto_fails_when_cuda_unavailable(self) -> None:
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="strict-gpu", device="auto"),
            ),
            asr_backend=StubBackend(),
        )
        with patch("whisper_omega.runtime.service.effective_device", return_value="cpu"):
            result = service.transcribe(__file__)

        self.assertEqual(result.status, "failure")
        self.assertEqual(result.error_code, "GPU_UNAVAILABLE")

    def test_permissive_alignment_failure_records_alignment_metadata(self) -> None:
        class FailingAlignmentBackend(UnavailableWav2Vec2Backend):
            def align(self, audio_path, text, segments, words, language):
                _ = (audio_path, text, segments, words, language)
                return AlignmentOutcome(
                    words=[],
                    strategy="romanized:ru",
                    token_source="text_map",
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

        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                required_features=["alignment"],
                align_backend="wav2vec2",
            ),
            asr_backend=StubBackend(),
            alignment_backend=FailingAlignmentBackend(),
        )
        result = service.transcribe(__file__)

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.metadata.alignment_strategy, "romanized:ru")
        self.assertEqual(result.metadata.alignment_token_source, "text_map")


if __name__ == "__main__":
    unittest.main()
