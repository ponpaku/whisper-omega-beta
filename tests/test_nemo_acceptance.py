from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_nemo_acceptance.py"
SPEC = importlib.util.spec_from_file_location("run_nemo_acceptance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_nemo_acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_nemo_acceptance)

_resolve_nemo_acceptance_audio = run_nemo_acceptance._resolve_nemo_acceptance_audio
build_nemo_acceptance_report = run_nemo_acceptance.build_nemo_acceptance_report


class NemoAcceptanceTests(unittest.TestCase):
    def test_resolve_nemo_acceptance_audio_prefers_fixture(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.is_file", return_value=True):
                resolved = _resolve_nemo_acceptance_audio(tmpdir)

        self.assertTrue(str(resolved).endswith("fixtures/d4_diarization/d4_mix_01.wav"))

    def test_report_blocks_when_nemo_is_missing(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"nemo": {"installed": False}},
            },
        }

        with patch.object(run_nemo_acceptance.DoctorReport, "collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
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

        with patch.object(run_nemo_acceptance.DoctorReport, "collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch.object(run_nemo_acceptance, "transcribe_file", side_effect=results):
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

        with patch.object(run_nemo_acceptance.DoctorReport, "collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch.object(run_nemo_acceptance, "transcribe_file", side_effect=results):
                report = build_nemo_acceptance_report()

        self.assertFalse(report["blocked"])
        self.assertFalse(report["all_passed"])
        self.assertEqual(report["cases"][1]["error_code"], "NEMO_MODEL_UNAVAILABLE")

    def test_report_forces_cpu_only_env_for_nemo_cases(self) -> None:
        doctor = {
            "faster_whisper_available": True,
            "backend_statuses": {
                "diarization": {"nemo": {"installed": True}},
            },
        }
        observed_envs: list[tuple[str | None, str | None]] = []

        def fake_transcribe_file(**kwargs):
            observed_envs.append(
                (
                    run_nemo_acceptance.os.environ.get("CUDA_VISIBLE_DEVICES"),
                    run_nemo_acceptance.os.environ.get("OMEGA_DEVICE"),
                )
            )
            if len(observed_envs) == 1:
                return SimpleNamespace(status="failure", error_code="CONFIG_INVALID", error_category="configuration", speakers=[])
            return SimpleNamespace(status="success", error_code=None, error_category=None, speakers=[object()])

        with patch.object(run_nemo_acceptance.DoctorReport, "collect", return_value=SimpleNamespace(to_dict=lambda: doctor)):
            with patch.object(run_nemo_acceptance, "transcribe_file", side_effect=fake_transcribe_file):
                report = build_nemo_acceptance_report()

        self.assertTrue(report["all_passed"])
        self.assertEqual(observed_envs, [("", "cpu"), ("", "cpu")])


if __name__ == "__main__":
    unittest.main()
