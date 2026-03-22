from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_nemo_acceptance import build_nemo_acceptance_report


class NemoAcceptanceTests(unittest.TestCase):
    def test_report_blocks_when_nemo_is_missing(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"nemo": {"installed": False}},
            },
        }

        with patch("scripts.run_nemo_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            report = build_nemo_acceptance_report()

        self.assertTrue(report["blocked"])
        self.assertIn("DIARIZATION_BACKEND_UNAVAILABLE", report["blocked_reasons"])

    def test_report_passes_when_invalid_config_and_success_cases_match(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"nemo": {"installed": True}},
            },
        }
        results = [
            SimpleNamespace(status="failure", error_code="CONFIG_INVALID", error_category="configuration", speakers=[]),
            SimpleNamespace(status="success", error_code=None, error_category=None, speakers=[object()]),
        ]

        with patch("scripts.run_nemo_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_nemo_acceptance.transcribe_file", side_effect=results):
                report = build_nemo_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["cases"][0]["error_code"], "CONFIG_INVALID")
        self.assertEqual(report["cases"][1]["speaker_count"], 1)

    def test_report_fails_when_success_case_does_not_succeed(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"nemo": {"installed": True}},
            },
        }
        results = [
            SimpleNamespace(status="failure", error_code="CONFIG_INVALID", error_category="configuration", speakers=[]),
            SimpleNamespace(status="failure", error_code="NEMO_MODEL_UNAVAILABLE", error_category="backend", speakers=[]),
        ]

        with patch("scripts.run_nemo_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_nemo_acceptance.transcribe_file", side_effect=results):
                report = build_nemo_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertFalse(report["all_passed"])
        self.assertEqual(report["cases"][1]["error_code"], "NEMO_MODEL_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
