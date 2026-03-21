from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.benchmark_smoke import run_case


class BenchmarkSmokeTests(unittest.TestCase):
    @patch("scripts.benchmark_smoke.transcribe_file")
    def test_run_case_collects_status_and_error_codes(self, transcribe_mock) -> None:
        transcribe_mock.side_effect = [
            type("Result", (), {"status": "success", "error_code": None})(),
            type("Result", (), {"status": "failure", "error_code": "AUDIO_DECODE_FAILURE"})(),
        ]

        payload = run_case(audio_path="dummy.wav", device="cpu", repeats=2, model_name="tiny")

        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["statuses"], ["success", "failure"])
        self.assertEqual(payload["error_codes"], [None, "AUDIO_DECODE_FAILURE"])


if __name__ == "__main__":
    unittest.main()
