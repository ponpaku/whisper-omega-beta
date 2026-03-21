from __future__ import annotations

import tempfile
import unittest
import wave
import json
from pathlib import Path

from scripts.build_dataset_manifest import build_markdown, collect_files, duration_seconds


class ValidationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.audio_path = self.root / "sample.wav"
        with wave.open(str(self.audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)
        self.float_audio_path = self.root / "float.wav"
        self.float_audio_path.write_bytes(
            (
                b"RIFF"
                + (36 + 4 * 16000).to_bytes(4, "little")
                + b"WAVE"
                + b"fmt "
                + (16).to_bytes(4, "little")
                + (3).to_bytes(2, "little")
                + (1).to_bytes(2, "little")
                + (16000).to_bytes(4, "little")
                + (16000 * 4).to_bytes(4, "little")
                + (4).to_bytes(2, "little")
                + (32).to_bytes(2, "little")
                + b"data"
                + (4 * 16000).to_bytes(4, "little")
                + (b"\x00\x00\x00\x00" * 16000)
            )
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_collect_files_lists_audio_inputs(self) -> None:
        files = collect_files(self.root)
        self.assertEqual(files, [self.float_audio_path, self.audio_path])

    def test_duration_seconds_reads_wave_length(self) -> None:
        self.assertEqual(duration_seconds(self.audio_path), "1.000")

    def test_duration_seconds_reads_float_wave_length(self) -> None:
        self.assertEqual(duration_seconds(self.float_audio_path), "1.000")

    def test_build_markdown_emits_manifest_row(self) -> None:
        markdown = build_markdown(self.root, "D1_SHORT_EN")
        self.assertIn("| D1_SHORT_EN | sample.wav |", markdown)
        self.assertIn("|---|---|---|---|---|---|---|", markdown)

    def test_build_markdown_uses_manifest_and_recipe_metadata(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps([{"file": "sample.wav", "note": "fixture note", "expected_error_code": "AUDIO_DECODE_FAILURE"}]),
            encoding="utf-8",
        )
        (self.root / "sample.recipe.json").write_text(
            json.dumps({"inputs": [{"file": "a.wav"}, {"file": "b.wav"}], "gap_ms": 250}),
            encoding="utf-8",
        )

        markdown = build_markdown(self.root, "D3_LONG_MIXED")

        self.assertIn("| D3_LONG_MIXED | sample.wav |", markdown)
        self.assertIn("| ja+en | 1 | fixture note; expected=AUDIO_DECODE_FAILURE; concatenated_inputs=2; gap_ms=250 |", markdown)

    def test_build_markdown_uses_track_recipe_metadata(self) -> None:
        (self.root / "sample.recipe.json").write_text(
            json.dumps(
                {
                    "tracks": [
                        {"id": "SPEAKER_A", "offset_ms": 0},
                        {"id": "SPEAKER_B", "offset_ms": 300},
                        {"id": "SPEAKER_C", "offset_ms": 1800},
                    ]
                }
            ),
            encoding="utf-8",
        )

        markdown = build_markdown(self.root, "D4_DIARIZATION")

        self.assertIn("| D4_DIARIZATION | sample.wav |", markdown)
        self.assertIn("| mixed | 3 | track_offsets_ms=0,300,1800 |", markdown)


if __name__ == "__main__":
    unittest.main()
