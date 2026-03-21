from __future__ import annotations

import os
import unittest

from whisper_omega.diarize.base import UnavailablePyannoteBackend
from whisper_omega.runtime.models import Segment, Word


class DiarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_hf_token = os.environ.pop("HF_TOKEN", None)
        self.original_audio_path = os.environ.pop("OMEGA_AUDIO_PATH", None)

    def tearDown(self) -> None:
        if self.original_hf_token is not None:
            os.environ["HF_TOKEN"] = self.original_hf_token
        if self.original_audio_path is not None:
            os.environ["OMEGA_AUDIO_PATH"] = self.original_audio_path

    def test_missing_token_is_configuration_error(self) -> None:
        backend = UnavailablePyannoteBackend()
        outcome = backend.diarize(
            [Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
            [Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
        )
        self.assertIn(
            outcome.backend_errors[0].code,
            {"DIARIZATION_BACKEND_UNAVAILABLE", "HF_TOKEN_MISSING"},
        )


if __name__ == "__main__":
    unittest.main()
