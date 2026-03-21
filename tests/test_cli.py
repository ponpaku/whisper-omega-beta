from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.cli.main import main
from whisper_omega.runtime.models import Segment, Word


def extract_json(output: str) -> dict:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output.lstrip())
    return payload


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None):
        _ = (audio_path, model_name, device, batch_size)
        return BackendTranscription(
            text="hello world",
            language=language or "en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker=None)],
            words=[Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
        )


class MissingBackend(ASRBackend):
    name = "missing"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None):
        _ = (audio_path, model_name, language, device, batch_size)
        raise RuntimeError("DEPENDENCY_MISSING:test-backend")


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmpdir.name) / "sample.wav"
        self.audio_path.write_bytes(b"RIFF")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_transcribe_emits_json_and_exit_zero(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            backend = StubBackend()
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(ServiceConfig(policy=PolicyConfig(runtime_policy="permissive", device="cpu")), asr_backend=backend)
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["text"], "hello world")
        self.assertIn("transcription completed", result.output)

    def test_strict_alignment_succeeds_with_word_timestamps(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            backend = StubBackend()
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="strict", device="cpu"),
                    required_features=["alignment"],
                    align_backend="wav2vec2",
                ),
                asr_backend=backend,
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--runtime-policy",
                    "strict",
                    "--require-alignment",
                    "--align-backend",
                    "wav2vec2",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "success")
        self.assertIn("alignment", payload["metadata"]["completed_features"])

    def test_usage_error_returns_40(self) -> None:
        result = self.runner.invoke(
            main,
            [
                "transcribe",
                str(self.audio_path),
                "--runtime-policy",
                "strict-gpu",
                "--device",
                "cpu",
            ],
        )
        self.assertEqual(result.exit_code, 40)
        self.assertIn("usage error", result.output)

    def test_output_file_writes_srt(self) -> None:
        output_path = Path(self.tmpdir.name) / "out.srt"
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(policy=PolicyConfig(runtime_policy="permissive", device="cpu")),
                asr_backend=StubBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--output-format",
                    "srt",
                    "--output-file",
                    str(output_path),
                    "--emit-result-json",
                    "never",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(output_path.exists())
        self.assertIn("00:00:00,000 --> 00:00:01,000", output_path.read_text(encoding="utf-8"))

    def test_failure_json_written_only_with_flag(self) -> None:
        output_path = Path(self.tmpdir.name) / "failure.json"
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(policy=PolicyConfig(runtime_policy="permissive", device="cpu")),
                asr_backend=MissingBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--output-file",
                    str(output_path),
                    "--emit-result-json",
                    "never",
                ],
            )
            self.assertEqual(result.exit_code, 30)
            self.assertFalse(output_path.exists())

            service.config.write_failure_json = True
            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--output-file",
                    str(output_path),
                    "--write-failure-json",
                    "--emit-result-json",
                    "never",
                ],
            )
            self.assertEqual(result.exit_code, 30)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failure")

    def test_whisperx_frontend_maps_legacy_flags(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    required_features=["diarization"],
                    diarize_backend="pyannote",
                ),
                asr_backend=StubBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "whisperx",
                    str(self.audio_path),
                    "--device",
                    "cpu",
                    "--diarize",
                ],
            )

        self.assertEqual(result.exit_code, 10)
        self.assertIn("degraded:", result.output)


if __name__ == "__main__":
    unittest.main()
