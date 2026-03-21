from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import wave
import builtins
from pathlib import Path
from unittest import mock

from scripts.export_google_fleurs_fixtures import (
    _float_to_pcm16,
    _to_mono_samples,
    dataset_config,
    export_fixture,
    parse_repo_tsv,
    repo_asset_url,
)


class GoogleFleursExportTests(unittest.TestCase):
    def test_dataset_config_returns_known_google_fixture(self) -> None:
        cfg = dataset_config("D1_SHORT_JA")
        self.assertEqual(cfg["dataset"], "google/fleurs")
        self.assertEqual(cfg["config"], "ja_jp")

    def test_to_mono_samples_handles_stereo_lists(self) -> None:
        self.assertEqual(_to_mono_samples([[1.0, -1.0], [0.5, 0.5]]), [0.0, 0.5])

    def test_float_to_pcm16_clips_samples(self) -> None:
        payload = _float_to_pcm16([-2.0, 0.0, 2.0])
        self.assertEqual(len(payload), 6)

    def test_repo_asset_url_builds_huggingface_path(self) -> None:
        cfg = dataset_config("D1_SHORT_JA")
        self.assertIn("/data/ja_jp/dev.tsv", repo_asset_url(cfg, "dev.tsv"))

    def test_parse_repo_tsv_reads_transcription_rows(self) -> None:
        rows = parse_repo_tsv("123\tabc.wav\tこんにちは\n456\tdef.wav\thello\n", 1)
        self.assertEqual(rows, [{"id": "123", "file": "abc.wav", "transcription": "こんにちは"}])

    def test_export_fixture_falls_back_to_repo_archive(self) -> None:
        cfg = dataset_config("D1_SHORT_JA")
        tsv_text = "123\tclip.wav\tこんにちは\n"
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo(name="audio/clip.wav")
            payload = wav_buffer.getvalue()
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        def fake_urlopen(url: str):
            if url == repo_asset_url(cfg, "dev.tsv"):
                return FakeResponse(tsv_text.encode("utf-8"))
            if url == repo_asset_url(cfg, "audio/dev.tar.gz"):
                return FakeResponse(tar_buffer.getvalue())
            raise AssertionError(url)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                raise ImportError("datasets unavailable for test")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch("scripts.export_google_fleurs_fixtures.urlopen", side_effect=fake_urlopen):
                with mock.patch("builtins.__import__", side_effect=fake_import):
                    manifest = export_fixture("D1_SHORT_JA", output_dir, 1)
            self.assertEqual(manifest[0]["file"], "clip.wav")
            self.assertTrue((output_dir / "clip.wav").exists())


if __name__ == "__main__":
    unittest.main()
