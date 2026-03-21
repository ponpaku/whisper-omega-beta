from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import torch

from whisper_omega.diarize.base import UnavailablePyannoteBackend
from whisper_omega.runtime.models import Segment, Word


class DiarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_hf_token = os.environ.pop("HF_TOKEN", None)
        self.original_audio_path = os.environ.pop("OMEGA_AUDIO_PATH", None)
        self.original_num_speakers = os.environ.pop("OMEGA_PYANNOTE_NUM_SPEAKERS", None)
        self.original_min_speakers = os.environ.pop("OMEGA_PYANNOTE_MIN_SPEAKERS", None)
        self.original_max_speakers = os.environ.pop("OMEGA_PYANNOTE_MAX_SPEAKERS", None)

    def tearDown(self) -> None:
        if self.original_hf_token is not None:
            os.environ["HF_TOKEN"] = self.original_hf_token
        if self.original_audio_path is not None:
            os.environ["OMEGA_AUDIO_PATH"] = self.original_audio_path
        if self.original_num_speakers is not None:
            os.environ["OMEGA_PYANNOTE_NUM_SPEAKERS"] = self.original_num_speakers
        if self.original_min_speakers is not None:
            os.environ["OMEGA_PYANNOTE_MIN_SPEAKERS"] = self.original_min_speakers
        if self.original_max_speakers is not None:
            os.environ["OMEGA_PYANNOTE_MAX_SPEAKERS"] = self.original_max_speakers

    def test_missing_token_is_configuration_error(self) -> None:
        fake_pipeline = types.SimpleNamespace(Pipeline=object)
        backend = UnavailablePyannoteBackend()
        with patch.dict("sys.modules", {"pyannote.audio": fake_pipeline}):
            outcome = backend.diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )
        self.assertIn(
            outcome.backend_errors[0].code,
            {"DIARIZATION_BACKEND_UNAVAILABLE", "HF_TOKEN_MISSING"},
        )

    def test_diarize_prefers_in_memory_waveform_when_torchaudio_is_available(self) -> None:
        class FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class FakeDiarization:
            def itertracks(self, yield_label: bool = False):
                _ = yield_label
                yield (FakeTurn(0.0, 1.0), None, "SPEAKER_00")

        captured = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                return cls()

            def __call__(self, audio_input):
                captured["audio_input"] = audio_input
                return FakeDiarization()

        fake_torchaudio = types.SimpleNamespace(
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )

        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        with patch.dict(
            "sys.modules",
            {
                "pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline),
                "torchaudio": fake_torchaudio,
            },
        ):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertFalse(outcome.backend_errors)
        self.assertIsInstance(captured["audio_input"], dict)
        self.assertIn("waveform", captured["audio_input"])
        self.assertEqual(outcome.segments[0].speaker, "SPEAKER_00")

    def test_diarize_passes_speaker_hints_to_pipeline(self) -> None:
        class FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class FakeDiarization:
            def itertracks(self, yield_label: bool = False):
                _ = yield_label
                yield (FakeTurn(0.0, 1.0), None, "SPEAKER_00")

        captured = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                return cls()

            def __call__(self, audio_input, **kwargs):
                captured["audio_input"] = audio_input
                captured["kwargs"] = kwargs
                return FakeDiarization()

        fake_torchaudio = types.SimpleNamespace(
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )

        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        os.environ["OMEGA_PYANNOTE_MIN_SPEAKERS"] = "2"
        os.environ["OMEGA_PYANNOTE_MAX_SPEAKERS"] = "3"
        with patch.dict(
            "sys.modules",
            {
                "pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline),
                "torchaudio": fake_torchaudio,
            },
        ):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(captured["kwargs"], {"min_speakers": 2, "max_speakers": 3})

    def test_invalid_speaker_hint_is_configuration_error(self) -> None:
        fake_pipeline = types.SimpleNamespace(Pipeline=object)
        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        os.environ["OMEGA_PYANNOTE_NUM_SPEAKERS"] = "0"
        with patch.dict("sys.modules", {"pyannote.audio": fake_pipeline}):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertEqual(outcome.backend_errors[0].code, "CONFIG_INVALID")

    def test_invalid_token_is_auth_failure(self) -> None:
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                raise RuntimeError("401 Unauthorized")

        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        with patch.dict("sys.modules", {"pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline)}):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertEqual(outcome.backend_errors[0].code, "DIARIZATION_AUTH_FAILURE")
        self.assertEqual(outcome.backend_errors[0].category, "configuration")

    def test_missing_model_is_model_unavailable(self) -> None:
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                raise FileNotFoundError("model repository not found")

        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        with patch.dict("sys.modules", {"pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline)}):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertEqual(outcome.backend_errors[0].code, "DIARIZATION_MODEL_UNAVAILABLE")
        self.assertEqual(outcome.backend_errors[0].category, "backend")

    def test_decode_error_is_runtime_failure(self) -> None:
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                return cls()

            def __call__(self, audio_input, **kwargs):
                _ = (audio_input, kwargs)
                raise RuntimeError("ffmpeg decode error")

        os.environ["HF_TOKEN"] = "secret"
        os.environ["OMEGA_AUDIO_PATH"] = "dummy.wav"
        with patch.dict("sys.modules", {"pyannote.audio": types.SimpleNamespace(Pipeline=FakePipeline)}):
            outcome = UnavailablePyannoteBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
            )

        self.assertEqual(outcome.backend_errors[0].code, "DIARIZATION_DECODE_FAILURE")
        self.assertEqual(outcome.backend_errors[0].category, "runtime")


if __name__ == "__main__":
    unittest.main()
