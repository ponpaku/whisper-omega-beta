from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_pyannote_acceptance import build_pyannote_acceptance_report


class PyannoteAcceptanceTests(unittest.TestCase):
    def test_report_blocks_when_core_prerequisites_missing(self) -> None:
        doctor = {
            "faster_whisper_available": False,
            "backend_statuses": {
                "diarization": {"pyannote": {"installed": False}},
                "decode": {
                    "torchaudio": {"ready": False},
                    "ffmpeg_torchcodec": {"ready": False},
                },
            },
        }

        with patch("scripts.run_pyannote_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            report = build_pyannote_acceptance_report()

        self.assertTrue(report["blocked"])
        self.assertIn("DEPENDENCY_MISSING", report["blocked_reasons"])
        self.assertEqual(len(report["cases"]), 2)

    def test_report_records_missing_token_case_and_blocks_token_case_without_token(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"pyannote": {"installed": True}},
                "decode": {
                    "torchaudio": {"ready": True},
                    "ffmpeg_torchcodec": {"ready": False},
                },
            },
        }

        results = [
            SimpleNamespace(status="degraded", error_code="HF_TOKEN_MISSING", error_category="configuration", speakers=[]),
        ]

        with patch("scripts.run_pyannote_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_pyannote_acceptance.transcribe_file", side_effect=results):
                with patch.dict(os.environ, {}, clear=True):
                    report = build_pyannote_acceptance_report()

        self.assertTrue(report["blocked"])
        self.assertEqual(report["cases"][0]["error_code"], "HF_TOKEN_MISSING")
        self.assertTrue(report["cases"][0]["passed"])
        self.assertEqual(report["cases"][1]["blocked_reason"], "HF_TOKEN_MISSING")

    def test_report_passes_when_missing_token_case_and_success_case_match_expectations(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"pyannote": {"installed": True}},
                "decode": {
                    "torchaudio": {"ready": True},
                    "ffmpeg_torchcodec": {"ready": False},
                },
            },
        }

        results = [
            SimpleNamespace(status="degraded", error_code="HF_TOKEN_MISSING", error_category="configuration", speakers=[]),
            SimpleNamespace(status="success", error_code=None, error_category=None, speakers=[object(), object()]),
        ]

        with patch("scripts.run_pyannote_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_pyannote_acceptance.transcribe_file", side_effect=results):
                with patch.dict(os.environ, {"HF_TOKEN": "secret"}, clear=True):
                    report = build_pyannote_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["cases"][1]["speaker_count"], 2)


if __name__ == "__main__":
    unittest.main()
