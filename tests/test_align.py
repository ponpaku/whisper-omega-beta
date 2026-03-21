from __future__ import annotations

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from whisper_omega.align.base import Wav2Vec2AlignmentBackend
from whisper_omega.runtime.models import Segment, Word


class Span:
    def __init__(self, start: int, end: int, score: float) -> None:
        self.start = start
        self.end = end
        self.score = score


class FakeModel:
    def __call__(self, waveform):
        _ = waveform
        return torch.zeros((1, 20, 4)), None


class FakeBundle:
    sample_rate = 16000

    def get_model(self):
        return FakeModel()

    def get_tokenizer(self):
        return lambda words: [[1] * max(1, len(word)) for word in words]

    def get_aligner(self):
        return lambda emission, tokens: [
            [Span(index * 5, (index + 1) * 5, 0.8 + (index * 0.05))] for index, _ in enumerate(tokens)
        ]


class AlignTests(unittest.TestCase):
    def test_wav2vec2_backend_realigns_word_times(self) -> None:
        fake_torchaudio = types.SimpleNamespace(
            pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
            functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)

        backend = Wav2Vec2AlignmentBackend()
        with patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
            outcome = backend.align(
                audio_path=__file__,
                text="hello world",
                segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker=None)],
                words=[
                    Word(text="hello", start=0.0, end=0.1, speaker=None),
                    Word(text="world", start=0.1, end=0.2, speaker=None),
                ],
                language="en",
            )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(len(outcome.words), 2)
        self.assertAlmostEqual(outcome.words[0].start, 0.0, places=3)
        self.assertAlmostEqual(outcome.words[0].end, 0.25, places=3)
        self.assertAlmostEqual(outcome.words[1].start, 0.25, places=3)
        self.assertAlmostEqual(outcome.words[1].end, 0.5, places=3)

    def test_wav2vec2_backend_rejects_unsupported_text(self) -> None:
        fake_torchaudio = types.SimpleNamespace(
            pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
            functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)

        backend = Wav2Vec2AlignmentBackend()
        with patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
            outcome = backend.align(
                audio_path=__file__,
                text="12345",
                segments=[Segment(id=0, start=0.0, end=1.0, text="12345", speaker=None)],
                words=[],
                language="en",
            )

        self.assertEqual(outcome.backend_errors[0].code, "ALIGNMENT_TEXT_UNSUPPORTED")

    def test_wav2vec2_backend_supports_kana_japanese_without_external_romanizer(self) -> None:
        fake_torchaudio = types.SimpleNamespace(
            pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
            functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)

        backend = Wav2Vec2AlignmentBackend()
        with patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
            outcome = backend.align(
                audio_path=__file__,
                text="こんにちは",
                segments=[Segment(id=0, start=0.0, end=1.0, text="こんにちは", speaker=None)],
                words=[Word(text="こんにちは", start=0.0, end=0.5, speaker=None)],
                language="ja",
            )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(outcome.words[0].text, "こんにちは")

    def test_wav2vec2_backend_rejects_unsupported_ja_kanji_text(self) -> None:
        fake_torchaudio = types.SimpleNamespace(
            pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
            functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)

        backend = Wav2Vec2AlignmentBackend()
        with patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}):
            outcome = backend.align(
                audio_path=__file__,
                text="日本語",
                segments=[Segment(id=0, start=0.0, end=1.0, text="日本語", speaker=None)],
                words=[Word(text="日本語", start=0.0, end=0.5, speaker=None)],
                language="ja",
            )

        self.assertEqual(outcome.backend_errors[0].code, "ALIGNMENT_TEXT_UNSUPPORTED")

    def test_wav2vec2_backend_can_use_external_romanizer(self) -> None:
        fake_torchaudio = types.SimpleNamespace(
            pipelines=types.SimpleNamespace(MMS_FA=FakeBundle()),
            functional=types.SimpleNamespace(resample=lambda waveform, sample_rate, target_rate: waveform),
            load=lambda path: (torch.ones((1, 16000)), 16000),
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        completed = types.SimpleNamespace(stdout="konnichiwa", stderr="", returncode=0)

        backend = Wav2Vec2AlignmentBackend()
        with patch.dict(
            "os.environ",
            {"OMEGA_ALIGNMENT_ROMANIZER": "fake-romanizer"},
            clear=False,
        ), patch.dict("sys.modules", {"torchaudio": fake_torchaudio, "torch": fake_torch}), patch(
            "whisper_omega.align.base.subprocess.run",
            return_value=completed,
        ):
            outcome = backend.align(
                audio_path=__file__,
                text="こんにちは",
                segments=[Segment(id=0, start=0.0, end=1.0, text="こんにちは", speaker=None)],
                words=[Word(text="こんにちは", start=0.0, end=0.5, speaker=None)],
                language="ja",
            )

        self.assertFalse(outcome.backend_errors)
        self.assertEqual(outcome.words[0].text, "こんにちは")


if __name__ == "__main__":
    unittest.main()
