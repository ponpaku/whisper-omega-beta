from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from whisper_omega.align.base import UnavailableWav2Vec2Backend
from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.cli.main import main
from whisper_omega.compat.whisperx import map_whisperx_options
from whisper_omega.diarize.base import DiarizationBackend, DiarizationOutcome
from whisper_omega.runtime.models import BackendError, Segment, Speaker, Word


def extract_json(output: str) -> dict:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output.lstrip())
    return payload


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None, word_timestamps=True):
        _ = (audio_path, model_name, device, batch_size, word_timestamps)
        return BackendTranscription(
            text="hello world",
            language=language or "en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker=None)],
            words=[Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9)],
        )


class MissingBackend(ASRBackend):
    name = "missing"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None, word_timestamps=True):
        _ = (audio_path, model_name, language, device, batch_size, word_timestamps)
        raise RuntimeError("DEPENDENCY_MISSING:test-backend")


class MissingDiarizationBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments, words):
        return DiarizationOutcome(
            segments=segments,
            words=words,
            speakers=[],
            backend_errors=[
                BackendError(
                    backend=self.name,
                    code="HF_TOKEN_MISSING",
                    category="configuration",
                    message="HF_TOKEN is required",
                    retryable=False,
                )
            ],
        )


class ConfigDiarizationBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments, words):
        return DiarizationOutcome(
            segments=segments,
            words=words,
            speakers=[],
            backend_errors=[
                BackendError(
                    backend=self.name,
                    code="HF_TOKEN_MISSING",
                    category="configuration",
                    message="HF_TOKEN is required",
                    retryable=False,
                )
            ],
        )


class CompatibilityMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmpdir.name) / "sample.wav"
        self.audio_path.write_bytes(b"RIFF")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_cli_matrix_cases(self) -> None:
        from whisper_omega.runtime.policy import PolicyConfig
        from whisper_omega.runtime.service import ServiceConfig, TranscriptionService

        cases = [
            ("CLI-001", ["whisperx", str(self.audio_path), "--device", "cpu", "--model", "small"], 0, "transcription completed"),
            ("CLI-002", ["whisperx", str(self.audio_path), "--device", "cpu"], 0, "transcription completed"),
            ("CLI-003", ["whisperx", str(self.audio_path), "--device", "cpu", "--language", "ja"], 0, "transcription completed"),
            ("CLI-004", ["whisperx", str(self.audio_path), "--device", "cpu", "--batch_size", "4"], 0, "transcription completed"),
            ("CLI-005", ["whisperx", str(self.audio_path), "--device", "cpu", "--output_format", "json"], 0, "transcription completed"),
            ("CLI-006", ["whisperx", str(self.audio_path), "--device", "cpu", "--diarize"], 10, "degraded:"),
            ("CLI-007", ["whisperx", str(self.audio_path), "--device", "cpu", "--highlight_words"], 0, "--highlight_words is accepted for compatibility"),
            ("CLI-008", ["whisperx", str(self.audio_path), "--device", "cpu", "--diarize", "--hf_token", "secret-token"], 10, "degraded:"),
            ("CLI-009", ["whisperx", str(self.audio_path), "--device", "cpu", "--align_model", "ja"], 0, "--align_model=ja requests alignment"),
        ]

        for case_id, argv, expected_exit, expected_text in cases:
            with self.subTest(case_id=case_id):
                with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
                    service = service_cls.return_value
                    if case_id in {"CLI-006", "CLI-008"}:
                        real_service = TranscriptionService(
                            ServiceConfig(
                                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                                required_features=["diarization"],
                                diarize_backend="pyannote",
                            ),
                            asr_backend=StubBackend(),
                            diarization_backend=MissingDiarizationBackend(),
                        )
                    elif case_id == "CLI-009":
                        real_service = TranscriptionService(
                            ServiceConfig(
                                policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                                required_features=["alignment"],
                                align_backend="wav2vec2",
                            ),
                            asr_backend=StubBackend(),
                            alignment_backend=UnavailableWav2Vec2Backend(),
                        )
                    else:
                        real_service = TranscriptionService(
                            ServiceConfig(policy=PolicyConfig(runtime_policy="permissive", device="cpu")),
                            asr_backend=StubBackend(),
                        )
                    service.transcribe.side_effect = real_service.transcribe
                    service.write_output.side_effect = real_service.write_output
                    service.config = real_service.config
                    result = self.runner.invoke(main, argv)

                self.assertEqual(result.exit_code, expected_exit)
                self.assertIn(expected_text, result.output)

    def test_whisperx_diarize_stays_on_pyannote_backend(self) -> None:
        compat = map_whisperx_options(
            diarize=True,
            align_model=None,
            output_format="json",
            batch_size=None,
        )

        self.assertTrue(compat.require_diarization)
        self.assertEqual(compat.diarize_backend, "pyannote")

    def test_json_matrix_cases(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "status": "success",
            "text": "hello world",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world", "speaker": "SPEAKER_00"}],
            "words": [{"text": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00", "confidence": 0.9}],
            "speakers": [{"id": "SPEAKER_00", "start": 0.0, "end": 1.0, "label": "Speaker 0"}],
            "metadata": {
                "asr_backend": "stub",
                "align_backend": "wav2vec2",
                "diarization_backend": "pyannote",
                "device": "cpu",
                "requested_device": "cpu",
                "actual_device": "cpu",
                "alignment_strategy": "en",
                "alignment_token_source": "native",
                "fallbacks": [],
                "requested_features": ["asr", "alignment", "diarization"],
                "completed_features": ["asr", "alignment", "diarization"],
                "failed_features": [],
            },
            "error_code": None,
            "error_category": None,
            "backend_errors": [],
        }
        cases = {
            "JSON-001": lambda data: self.assertTrue(data["segments"]),
            "JSON-002": lambda data: self.assertTrue(data["words"]),
            "JSON-003": lambda data: self.assertIn("speaker", data["segments"][0]),
            "JSON-004": lambda data: self.assertIn("speaker", data["words"][0]),
            "JSON-005": lambda data: self.assertTrue(data["speakers"]),
            "JSON-006": lambda data: self.assertIn("metadata", data),
            "JSON-007": lambda data: self.assertEqual(data["status"], "success"),
            "JSON-008": lambda data: self.assertEqual(data["schema_version"], "1.0.0"),
        }
        for case_id, assertion in cases.items():
            with self.subTest(case_id=case_id):
                assertion(payload)

    def test_exit_matrix_cases(self) -> None:
        from whisper_omega.runtime.policy import PolicyConfig
        from whisper_omega.runtime.service import ServiceConfig, TranscriptionService

        cases = [
            ("EXIT-001", StubBackend(), {}, 0),
            ("EXIT-002", StubBackend(), {"required_features": ["diarization"], "diarize_backend": "pyannote", "diarization_backend": MissingDiarizationBackend()}, 10),
            ("EXIT-003", StubBackend(), {"runtime_policy": "strict-gpu", "device": "auto", "patch_cpu": True}, 20),
            ("EXIT-004", MissingBackend(), {}, 30),
            (
                "EXIT-005",
                StubBackend(),
                {
                    "runtime_policy": "strict",
                    "required_features": ["diarization"],
                    "diarize_backend": "pyannote",
                    "diarization_backend": ConfigDiarizationBackend(),
                    "emit_json": "always",
                },
                31,
            ),
            ("EXIT-006", StubBackend(), {"usage": True}, 40),
        ]

        for case_id, backend, options, expected_exit in cases:
            with self.subTest(case_id=case_id):
                if options.get("usage"):
                    result = self.runner.invoke(
                        main,
                        ["transcribe", str(self.audio_path), "--runtime-policy", "strict-gpu", "--device", "cpu"],
                    )
                    self.assertEqual(result.exit_code, expected_exit)
                    continue

                with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
                    service = service_cls.return_value
                    cfg = ServiceConfig(
                        policy=PolicyConfig(
                            runtime_policy=options.get("runtime_policy", "permissive"),
                            device=options.get("device", "cpu"),
                        ),
                        required_features=options.get("required_features", []),
                        diarize_backend=options.get("diarize_backend", "none"),
                    )
                    real_service = TranscriptionService(
                        cfg,
                        asr_backend=backend,
                        diarization_backend=options.get("diarization_backend"),
                    )
                    service.transcribe.side_effect = real_service.transcribe
                    service.write_output.side_effect = real_service.write_output
                    service.config = real_service.config
                    invoke_args = ["transcribe", str(self.audio_path), "--device", options.get("device", "cpu")]
                    if options.get("required_features"):
                        invoke_args.extend(["--require-diarization", "--diarize-backend", "pyannote"])
                    if options.get("emit_json"):
                        invoke_args.extend(["--emit-result-json", options["emit_json"]])
                    if options.get("patch_cpu"):
                        with patch("whisper_omega.runtime.service.effective_device", return_value="cpu"):
                            result = self.runner.invoke(main, invoke_args)
                    else:
                        result = self.runner.invoke(main, invoke_args)

                self.assertEqual(result.exit_code, expected_exit)


if __name__ == "__main__":
    unittest.main()
