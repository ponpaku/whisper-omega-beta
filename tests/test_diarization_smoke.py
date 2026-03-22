from __future__ import annotations

import unittest

from scripts.run_diarization_smoke import build_diarization_smoke_report


class DiarizationSmokeTests(unittest.TestCase):
    def test_build_diarization_smoke_report_uses_d4_fixtures(self) -> None:
        report = build_diarization_smoke_report()

        self.assertTrue(report["all_passed"])
        self.assertEqual(len(report["checks"]), 2)
        for check in report["checks"]:
            self.assertEqual(len(check["backends"]), 2)
            self.assertEqual({item["backend"] for item in check["backends"]}, {"pyannote", "nemo"})
            for backend_check in check["backends"]:
                self.assertEqual(len(backend_check["speaker_ids"]), check["track_count"])
                self.assertEqual(backend_check["speaker_ids"], backend_check["assigned_speakers"])


if __name__ == "__main__":
    unittest.main()
