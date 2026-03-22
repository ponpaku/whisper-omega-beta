from __future__ import annotations

import unittest

from scripts.run_acceptance import acceptance_failures


class AcceptanceTests(unittest.TestCase):
    def test_acceptance_failures_empty_when_all_selected_steps_pass(self) -> None:
        report = {
            "doctor": {"returncode": 0},
            "tests": {"returncode": 0},
            "smoke": {"returncode": 0, "payload": None},
            "alignment_smoke": {"returncode": 0, "payload": {"all_passed": True}},
            "diarization_smoke": {"returncode": 0, "payload": {"all_passed": True}},
        }

        self.assertEqual(acceptance_failures(report), [])

    def test_acceptance_failures_collects_nonzero_and_failed_payloads(self) -> None:
        report = {
            "doctor": {"returncode": 0},
            "tests": {"returncode": 1},
            "smoke": {"returncode": 0, "payload": None},
            "alignment_smoke": {"returncode": 0, "payload": {"all_passed": False}},
            "diarization_smoke": {"returncode": 2, "payload": None},
        }

        self.assertEqual(
            acceptance_failures(report),
            ["tests", "alignment_smoke", "diarization_smoke"],
        )


if __name__ == "__main__":
    unittest.main()
