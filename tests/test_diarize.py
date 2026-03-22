from __future__ import annotations

import os
import types
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from whisper_omega.diarize.base import (
    ChannelDiarizationBackend,
    NemoDiarizationBackend,
    UnavailablePyannoteBackend,
    _build_nemo_config,
)
from whisper_omega.runtime.models import Segment, Word


class DiarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_hf_token = os.environ.pop("HF_TOKEN", None)
        self.original_audio_path = os.environ.pop("OMEGA_AUDIO_PATH", None)
        self.original_num_speakers = os.environ.pop("OMEGA_PYANNOTE_NUM_SPEAKERS", None)
        self.original_min_speakers = os.environ.pop("OMEGA_PYANNOTE_MIN_SPEAKERS", None)
        self.original_max_speakers = os.environ.pop("OMEGA_PYANNOTE_MAX_SPEAKERS", None)
        self.tmpdir = TemporaryDirectory()

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
        self.tmpdir.cleanup()

    def _write_stereo_wav(self, name: str) -> Path:
        path = Path(self.tmpdir.name) / name
        sample_rate = 8000
        frames = bytearray()
        for index in range(sample_rate):
            if index < sample_rate // 2:
                left, right = 20000, 500
            else:
                left, right = 500, 20000
            frames.extend(int(left).to_bytes(2, byteorder="little", signed=True))
            frames.extend(int(right).to_bytes(2, byteorder="little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frames))
        return path

    def _write_mono_wav(self, name: str) -> Path:
        path = Path(self.tmpdir.name) / name
        sample_rate = 8000
        frames = bytearray()
        for _ in range(sample_rate):
            frames.extend(int(12000).to_bytes(2, byteorder="little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frames))
        return path

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

    def test_diarize_supports_diarize_output_wrapper(self) -> None:
        class FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class FakeAnnotation:
            def itertracks(self, yield_label: bool = False):
                _ = yield_label
                yield (FakeTurn(0.0, 1.0), None, "SPEAKER_00")

        class FakeDiarizeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = FakeAnnotation()

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, token=None):
                _ = (model_id, token)
                return cls()

            def __call__(self, audio_input, **kwargs):
                _ = (audio_input, kwargs)
                return FakeDiarizeOutput()

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
        self.assertEqual(outcome.segments[0].speaker, "SPEAKER_00")

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

    def test_channel_backend_assigns_left_and_right_speakers(self) -> None:
        os.environ["OMEGA_AUDIO_PATH"] = str(self._write_stereo_wav("stereo.wav"))

        outcome = ChannelDiarizationBackend().diarize(
            [
                Segment(id=0, start=0.0, end=0.45, text="left", speaker=None),
                Segment(id=1, start=0.55, end=1.0, text="right", speaker=None),
            ],
            [
                Word(text="left", start=0.0, end=0.45, speaker=None, confidence=0.9),
                Word(text="right", start=0.55, end=1.0, speaker=None, confidence=0.9),
            ],
        )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(outcome.segments[0].speaker, "CHANNEL_LEFT")
        self.assertEqual(outcome.segments[1].speaker, "CHANNEL_RIGHT")
        self.assertEqual({speaker.id for speaker in outcome.speakers}, {"CHANNEL_LEFT", "CHANNEL_RIGHT"})

    def test_channel_backend_rejects_mono_audio(self) -> None:
        os.environ["OMEGA_AUDIO_PATH"] = str(self._write_mono_wav("mono.wav"))

        outcome = ChannelDiarizationBackend().diarize(
            [Segment(id=0, start=0.0, end=1.0, text="mono", speaker=None)],
            [Word(text="mono", start=0.0, end=1.0, speaker=None, confidence=0.9)],
        )

        self.assertEqual(outcome.backend_errors[0].code, "DIARIZATION_CHANNELS_UNAVAILABLE")
        self.assertEqual(outcome.backend_errors[0].category, "validation")

    def test_nemo_backend_assigns_speakers_from_rttm(self) -> None:
        audio_path = self._write_stereo_wav("nemo.wav")
        os.environ["OMEGA_AUDIO_PATH"] = str(audio_path)

        class FakeOmegaConf:
            @staticmethod
            def create(data):
                return data

            @staticmethod
            def merge(base, override):
                merged = dict(base)
                merged.update(override)
                return merged

            @staticmethod
            def load(path):
                _ = path
                return {}

        class FakeDiarizer:
            def __init__(self, cfg) -> None:
                self.cfg = cfg

            def diarize(self, paths2audio_files=None, batch_size=1):
                _ = (paths2audio_files, batch_size)
                out_dir = Path(self.cfg["diarizer"]["out_dir"]) / "pred_rttms"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "nemo.rttm").write_text(
                    "\n".join(
                        [
                            "SPEAKER nemo 1 0.000 0.450 <NA> <NA> SPEAKER_00 <NA> <NA>",
                            "SPEAKER nemo 1 0.550 0.450 <NA> <NA> SPEAKER_01 <NA> <NA>",
                        ]
                    ),
                    encoding="utf-8",
                )

        with patch.dict(
            "sys.modules",
            {
                "nemo.collections.asr.models": types.SimpleNamespace(ClusteringDiarizer=FakeDiarizer),
                "omegaconf": types.SimpleNamespace(OmegaConf=FakeOmegaConf),
            },
        ):
            outcome = NemoDiarizationBackend().diarize(
                [
                    Segment(id=0, start=0.0, end=0.45, text="left", speaker=None),
                    Segment(id=1, start=0.55, end=1.0, text="right", speaker=None),
                ],
                [
                    Word(text="left", start=0.0, end=0.45, speaker=None, confidence=0.9),
                    Word(text="right", start=0.55, end=1.0, speaker=None, confidence=0.9),
                ],
            )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(outcome.segments[0].speaker, "SPEAKER_00")
        self.assertEqual(outcome.segments[1].speaker, "SPEAKER_01")

    def test_nemo_backend_reports_missing_output(self) -> None:
        audio_path = self._write_stereo_wav("missing-output.wav")
        os.environ["OMEGA_AUDIO_PATH"] = str(audio_path)

        class FakeOmegaConf:
            @staticmethod
            def create(data):
                return data

            @staticmethod
            def merge(base, override):
                merged = dict(base)
                merged.update(override)
                return merged

            @staticmethod
            def load(path):
                _ = path
                return {}

        class FakeDiarizer:
            def __init__(self, cfg) -> None:
                self.cfg = cfg

            def diarize(self, paths2audio_files=None, batch_size=1):
                _ = (paths2audio_files, batch_size)
                return None

        with patch.dict(
            "sys.modules",
            {
                "nemo.collections.asr.models": types.SimpleNamespace(ClusteringDiarizer=FakeDiarizer),
                "omegaconf": types.SimpleNamespace(OmegaConf=FakeOmegaConf),
            },
        ):
            outcome = NemoDiarizationBackend().diarize(
                [Segment(id=0, start=0.0, end=1.0, text="mono", speaker=None)],
                [Word(text="mono", start=0.0, end=1.0, speaker=None, confidence=0.9)],
            )

        self.assertEqual(outcome.backend_errors[0].code, "NEMO_OUTPUT_MISSING")

    def test_nemo_config_includes_runtime_defaults(self) -> None:
        class FakeOmegaConf:
            @staticmethod
            def create(data):
                return data

            @staticmethod
            def merge(base, override):
                merged = dict(base)
                merged.update(override)
                return merged

            @staticmethod
            def load(path):
                _ = path
                return {}

        config = _build_nemo_config(FakeOmegaConf, "manifest.json", "out", device="cpu")

        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["sample_rate"], 16000)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["num_workers"], 0)
        self.assertFalse(config["verbose"])
        self.assertFalse(config["diarizer"]["clustering"]["parameters"]["oracle_num_speakers"])
        self.assertEqual(config["diarizer"]["clustering"]["parameters"]["max_num_speakers"], 8)
        self.assertEqual(config["diarizer"]["clustering"]["parameters"]["max_rp_threshold"], 0.25)
        self.assertEqual(config["diarizer"]["clustering"]["parameters"]["sparse_search_volume"], 30)
        self.assertEqual(config["diarizer"]["vad"]["parameters"]["onset"], 0.5)
        self.assertEqual(config["diarizer"]["vad"]["parameters"]["offset"], 0.5)
        self.assertTrue(config["diarizer"]["vad"]["parameters"]["filter_speech_first"])


if __name__ == "__main__":
    unittest.main()
