from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.generate_validation_report import build_report


class ValidationReportTests(unittest.TestCase):
    @patch("scripts.generate_validation_report.platform.platform", return_value="test-platform")
    @patch("scripts.generate_validation_report.subprocess.run")
    def test_build_report_collects_doctor_and_tests(self, run_mock, _platform_mock) -> None:
        run_mock.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": '{"status":"ok"}', "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": "tests ok", "stderr": ""})(),
        ]

        report = build_report(include_smoke=False)

        self.assertEqual(report["doctor"]["payload"], {"status": "ok"})
        self.assertEqual(report["tests"]["returncode"], 0)
        self.assertIsNone(report["smoke"])
        self.assertIsNone(report["alignment_smoke"])
        self.assertIsNone(report["diarization_smoke"])

    @patch("scripts.generate_validation_report.platform.platform", return_value="test-platform")
    @patch("scripts.generate_validation_report.subprocess.run")
    def test_build_report_can_include_smoke(self, run_mock, _platform_mock) -> None:
        run_mock.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": '{"status":"ok"}', "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": "tests ok", "stderr": ""})(),
            type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "smoke failed"})(),
        ]

        report = build_report(include_smoke=True)

        self.assertEqual(report["smoke"]["returncode"], 1)
        self.assertEqual(report["smoke"]["stderr"], "smoke failed")

    @patch("scripts.generate_validation_report.platform.platform", return_value="test-platform")
    @patch("scripts.generate_validation_report.subprocess.run")
    def test_build_report_can_include_alignment_smoke(self, run_mock, _platform_mock) -> None:
        run_mock.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": '{"status":"ok"}', "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": "tests ok", "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": '{"all_passed": true}', "stderr": ""})(),
        ]

        report = build_report(include_smoke=False, include_alignment_smoke=True)

        self.assertEqual(report["alignment_smoke"]["returncode"], 0)
        self.assertEqual(report["alignment_smoke"]["payload"], {"all_passed": True})

    @patch("scripts.generate_validation_report.platform.platform", return_value="test-platform")
    @patch("scripts.generate_validation_report.subprocess.run")
    def test_build_report_can_include_diarization_smoke(self, run_mock, _platform_mock) -> None:
        run_mock.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": '{"status":"ok"}', "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": "tests ok", "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": '{"all_passed": true}', "stderr": ""})(),
        ]

        report = build_report(include_smoke=False, include_diarization_smoke=True)

        self.assertEqual(report["diarization_smoke"]["returncode"], 0)
        self.assertEqual(report["diarization_smoke"]["payload"], {"all_passed": True})


if __name__ == "__main__":
    unittest.main()
