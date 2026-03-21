from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.build_long_fixture import concatenate_wavs
from scripts.build_diarization_fixture import build_mixture, build_multispeaker_mixture, read_wav_mono
from scripts.build_failure_fixtures import build_failure_fixtures


class FixtureBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.speaker_a = self.root / "speaker_a.wav"
        self.speaker_b = self.root / "speaker_b.wav"
        self.speaker_c = self.root / "speaker_c.wav"
        self._write_pcm16(self.speaker_a, sample_rate=16000, frames=1600, value=0.25)
        self._write_pcm16(self.speaker_b, sample_rate=16000, frames=1600, value=-0.25)
        self._write_pcm16(self.speaker_c, sample_rate=16000, frames=1600, value=0.5)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_pcm16(self, path: Path, sample_rate: int, frames: int, value: float) -> None:
        sample = int(max(-1.0, min(1.0, value)) * 32767.0)
        payload = sample.to_bytes(2, "little", signed=True) * frames
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(payload)

    def test_build_mixture_writes_output_and_recipe(self) -> None:
        output_path = self.root / "mix.wav"
        recipe = build_mixture(self.speaker_a, self.speaker_b, output_path, offset_ms=100, gain_a=1.0, gain_b=1.0)

        self.assertEqual(recipe["output_file"], "mix.wav")
        self.assertTrue(output_path.exists())
        self.assertTrue((self.root / "mix.recipe.json").exists())
        sample_rate, samples = read_wav_mono(output_path)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(samples), 3200)

    def test_build_failure_fixtures_creates_expected_cases(self) -> None:
        output_dir = self.root / "failures"
        manifest = build_failure_fixtures(self.speaker_a, output_dir)

        self.assertEqual(len(manifest), 4)
        self.assertEqual(manifest[-1]["file"], "missing.wav")
        self.assertEqual((output_dir / "empty.wav").read_bytes(), b"")
        saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest[0]["expected_error_code"], "AUDIO_DECODE_FAILURE")

    def test_concatenate_wavs_writes_recipe(self) -> None:
        output_path = self.root / "long.wav"
        recipe = concatenate_wavs([self.speaker_a, self.speaker_b], output_path, gap_ms=250)

        self.assertEqual(recipe["output_file"], "long.wav")
        self.assertEqual(len(recipe["inputs"]), 2)
        self.assertTrue((self.root / "long.recipe.json").exists())
        with wave.open(str(output_path), "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnframes(), 1600 + 1600 + 4000)

    def test_build_multispeaker_mixture_supports_three_speakers(self) -> None:
        output_path = self.root / "mix_three.wav"
        recipe = build_multispeaker_mixture(
            [
                {"id": "SPEAKER_A", "source": self.speaker_a, "offset_ms": 0, "gain": 1.0},
                {"id": "SPEAKER_B", "source": self.speaker_b, "offset_ms": 100, "gain": 1.0},
                {"id": "SPEAKER_C", "source": self.speaker_c, "offset_ms": 200, "gain": 1.0},
            ],
            output_path,
        )

        self.assertEqual(len(recipe["tracks"]), 3)
        self.assertEqual(recipe["tracks"][2]["id"], "SPEAKER_C")
        self.assertTrue((self.root / "mix_three.recipe.json").exists())


if __name__ == "__main__":
    unittest.main()
