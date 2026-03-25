from __future__ import annotations

import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from whisper_omega.align.base import AlignmentOutcome, UnavailableWav2Vec2Backend
from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.diarize.base import DiarizationBackend, DiarizationOutcome
from whisper_omega.runtime.models import BackendError, Segment, Word
from whisper_omega.runtime.policy import PolicyConfig
from whisper_omega.runtime.service import ServiceConfig, TranscriptionService


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None, word_timestamps=True):
        _ = (audio_path, model_name, language, device, batch_size, word_timestamps)
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
    def _write_wav(self, duration_ms: int = 1000) -> Path:
        tmpdir = TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "sample.wav"
        sample_rate = 8000
        frame_count = int(sample_rate * (duration_ms / 1000))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * frame_count)
        return path

    def test_permissive_optional_failures_become_degraded(self) -> None:
        audio_path = self._write_wav(1000)
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                required_features=["diarization"],
                diarize_backend="pyannote",
            ),
            asr_backend=StubBackend(),
            diarization_backend=MissingDiarizationBackend(),
        )
        result = service.transcribe(audio_path)
        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.metadata.failed_features, ["diarization"])
        self.assertEqual(len(result.backend_errors), 1)
        self.assertGreaterEqual(result.metadata.timings.total_ms, result.metadata.timings.asr_ms)
        self.assertGreaterEqual(result.metadata.timings.diarization_ms, 0)
        self.assertEqual(result.metadata.timings.audio_duration_ms, 1000)
        self.assertIsNotNone(result.metadata.timings.real_time_factor)

    def test_strict_alignment_succeeds_with_existing_words(self) -> None:
        audio_path = self._write_wav(1000)
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="strict", device="cpu"),
                required_features=["alignment"],
                align_backend="wav2vec2",
            ),
            asr_backend=StubBackend(),
            alignment_backend=UnavailableWav2Vec2Backend(),
        )
        result = service.transcribe(audio_path)
        self.assertEqual(result.status, "success")
        self.assertIn("alignment", result.metadata.completed_features)
        self.assertEqual(result.metadata.alignment_strategy, "fallback-existing-words")
        self.assertEqual(result.metadata.alignment_token_source, "asr_words")
        self.assertGreaterEqual(result.metadata.timings.alignment_ms, 0)

    def test_strict_gpu_with_auto_fails_when_cuda_unavailable(self) -> None:
        audio_path = self._write_wav(1000)
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="strict-gpu", device="auto"),
            ),
            asr_backend=StubBackend(),
        )
        with patch("whisper_omega.runtime.service.effective_device", return_value="cpu"):
            result = service.transcribe(audio_path)

        self.assertEqual(result.status, "failure")
        self.assertEqual(result.error_code, "GPU_UNAVAILABLE")
        self.assertGreaterEqual(result.metadata.timings.total_ms, 0)
        self.assertEqual(result.metadata.timings.audio_duration_ms, 1000)
        self.assertIsNotNone(result.metadata.timings.real_time_factor)

    def test_permissive_alignment_failure_records_alignment_metadata(self) -> None:
        audio_path = self._write_wav(1000)
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
        result = service.transcribe(audio_path)

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.metadata.alignment_strategy, "romanized:ru")
        self.assertEqual(result.metadata.alignment_token_source, "text_map")
        self.assertGreaterEqual(result.metadata.timings.alignment_ms, 0)

    def test_permissive_aligned_timestamp_strategy_degrades_to_direct(self) -> None:
        audio_path = self._write_wav(1000)

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
                align_backend="wav2vec2",
                timestamp_strategy="aligned",
            ),
            asr_backend=StubBackend(),
            alignment_backend=FailingAlignmentBackend(),
        )
        result = service.transcribe(audio_path)

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.metadata.requested_timestamp_strategy, "aligned")
        self.assertEqual(result.metadata.timestamp_strategy, "direct")
        self.assertEqual(result.metadata.timestamp_source, "direct")
        self.assertEqual(result.metadata.timestamp_quality, "exact")
        self.assertEqual(result.metadata.failed_features, [])
        self.assertEqual(result.metadata.fallbacks[0].type, "quality")
        self.assertEqual(result.metadata.fallbacks[0].from_value, "aligned")
        self.assertEqual(result.metadata.fallbacks[0].to_value, "direct")

    def test_hybrid_timestamp_strategy_skips_alignment_for_confident_words(self) -> None:
        audio_path = self._write_wav(1000)

        class ExplodingAlignmentBackend(UnavailableWav2Vec2Backend):
            def align(self, audio_path, text, segments, words, language):
                raise AssertionError("alignment should not run for confident hybrid output")

        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                align_backend="wav2vec2",
                timestamp_strategy="hybrid_selective_align",
            ),
            asr_backend=StubBackend(),
            alignment_backend=ExplodingAlignmentBackend(),
        )
        result = service.transcribe(audio_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata.timestamp_strategy, "hybrid_selective_align")
        self.assertEqual(result.metadata.timestamp_source, "direct")
        self.assertEqual(result.metadata.alignment_applied_segments, 0)
        self.assertEqual(result.metadata.alignment_skipped_segments, 1)


if __name__ == "__main__":
    unittest.main()
