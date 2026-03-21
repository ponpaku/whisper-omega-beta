from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_ja_reading_map import build_stub_map, extract_kanji_tokens


class JapaneseReadingMapTests(unittest.TestCase):
    def test_extract_kanji_tokens_finds_unique_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {"transcription": "日本語を話します。"},
                        {"transcription": "日本語の島々です。"},
                        {"transcription": "こんにちは"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            tokens = extract_kanji_tokens(manifest_path)

        self.assertEqual(tokens, ["島々です", "日本語", "話します"])

    def test_build_stub_map_initializes_empty_readings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps([{"transcription": "日本語を話します。"}], ensure_ascii=False),
                encoding="utf-8",
            )

            payload = build_stub_map(manifest_path)

        self.assertEqual(payload, {"日本語": "", "話します": ""})


if __name__ == "__main__":
    unittest.main()
