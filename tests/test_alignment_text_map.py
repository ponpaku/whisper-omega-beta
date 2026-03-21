from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_alignment_text_map import build_stub_map, extract_non_latin_tokens


class AlignmentTextMapTests(unittest.TestCase):
    def test_extract_non_latin_tokens_finds_unique_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {"transcription": "Привет мир"},
                        {"transcription": "こんにちは 世界"},
                        {"transcription": "hello world"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            tokens = extract_non_latin_tokens(manifest_path)

        self.assertEqual(tokens, ["Привет", "мир", "こんにちは", "世界"])

    def test_build_stub_map_initializes_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps([{"transcription": "Привет мир"}], ensure_ascii=False),
                encoding="utf-8",
            )

            payload = build_stub_map(manifest_path)

        self.assertEqual(payload, {"Привет": "", "мир": ""})


if __name__ == "__main__":
    unittest.main()
