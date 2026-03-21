from __future__ import annotations

import json
import unittest
from pathlib import Path

from whisper_omega.diarize.base import _speaker_for_interval


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "diarization_overlap_case.json"


class DiarizeContractTests(unittest.TestCase):
    def test_overlap_fixture_assigns_best_overlap_speaker(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        speaker_turns = [
            (item["start"], item["end"], item["speaker"])
            for item in payload["speaker_turns"]
        ]

        for segment in payload["segments"]:
            with self.subTest(kind="segment", start=segment["start"], end=segment["end"]):
                speaker = _speaker_for_interval(segment["start"], segment["end"], speaker_turns)
                self.assertEqual(speaker, segment["expected_speaker"])

        for word in payload["words"]:
            with self.subTest(kind="word", start=word["start"], end=word["end"]):
                speaker = _speaker_for_interval(word["start"], word["end"], speaker_turns)
                self.assertEqual(speaker, word["expected_speaker"])


if __name__ == "__main__":
    unittest.main()
