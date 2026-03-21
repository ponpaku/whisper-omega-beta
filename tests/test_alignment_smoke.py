from __future__ import annotations

import unittest

from scripts.run_alignment_smoke import build_alignment_smoke_report


class AlignmentSmokeTests(unittest.TestCase):
    def test_build_alignment_smoke_report_uses_fixture_manifests(self) -> None:
        report = build_alignment_smoke_report()

        self.assertTrue(report["all_passed"])
        self.assertEqual(len(report["checks"]), 2)

        d2_check = next(item for item in report["checks"] if item["dataset"] == "D2_SHORT_EN")
        self.assertEqual(d2_check["strategy"], "en")
        self.assertEqual(d2_check["token_source"], "native")
        self.assertGreater(d2_check["word_count"], 1)

        d1_check = next(item for item in report["checks"] if item["dataset"] == "D1_SHORT_JA")
        self.assertEqual(d1_check["strategy"], "ja-kana")
        self.assertEqual(d1_check["token_source"], "ja_reading_map")
        self.assertEqual(d1_check["word_count"], 1)


if __name__ == "__main__":
    unittest.main()
