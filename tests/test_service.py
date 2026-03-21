from __future__ import annotations

import unittest

from whisper_omega.align.base import UnavailableWav2Vec2Backend
from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.diarize.base import UnavailablePyannoteBackend
from whisper_omega.runtime.models import Segment, Word
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


class ServiceTests(unittest.TestCase):
    def test_permissive_optional_failures_become_degraded(self) -> None:
        service = TranscriptionService(
            ServiceConfig(
                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                required_features=["diarization"],
                diarize_backend="pyannote",
            ),
            asr_backend=StubBackend(),
            diarization_backend=UnavailablePyannoteBackend(),
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


if __name__ == "__main__":
    unittest.main()
