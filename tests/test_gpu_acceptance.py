from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_gpu_acceptance import build_gpu_acceptance_report


def _result(requested_device: str, actual_device: str, status: str, error_code: str | None):
    return SimpleNamespace(
        status=status,
        error_code=error_code,
        error_category="runtime" if error_code else None,
        metadata=SimpleNamespace(requested_device=requested_device, actual_device=actual_device),
    )


class GpuAcceptanceTests(unittest.TestCase):
    def test_report_blocks_when_cuda_unavailable(self) -> None:
        doctor = {"faster_whisper_available": True, "torch_cuda_available": False}

        with patch("scripts.run_gpu_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_gpu_acceptance.cuda_available", return_value=False):
                report = build_gpu_acceptance_report()

        self.assertTrue(report["blocked"])
        self.assertIn("GPU_UNAVAILABLE", report["blocked_reasons"])

    def test_report_passes_when_cuda_cases_use_gpu(self) -> None:
        doctor = {"faster_whisper_available": True, "torch_cuda_available": True}
        results = [
            _result("auto", "cuda", "success", None),
            _result("cuda", "cuda", "success", None),
            _result("auto", "cuda", "success", None),
        ]

        with patch("scripts.run_gpu_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_gpu_acceptance.cuda_available", return_value=True):
                with patch("scripts.run_gpu_acceptance.transcribe_file", side_effect=results):
                    report = build_gpu_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["residual_risks"], [])

    def test_report_records_decode_failure_as_residual_risk(self) -> None:
        doctor = {"faster_whisper_available": True, "torch_cuda_available": True}
        results = [
            _result("auto", "cuda", "failure", "AUDIO_DECODE_FAILURE"),
            _result("cuda", "cuda", "failure", "AUDIO_DECODE_FAILURE"),
            _result("auto", "cuda", "failure", "AUDIO_DECODE_FAILURE"),
        ]

        with patch("scripts.run_gpu_acceptance.DoctorReport.collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch("scripts.run_gpu_acceptance.cuda_available", return_value=True):
                with patch("scripts.run_gpu_acceptance.transcribe_file", side_effect=results):
                    report = build_gpu_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertTrue(report["all_passed"])
        self.assertEqual(len(report["residual_risks"]), 3)


if __name__ == "__main__":
    unittest.main()
