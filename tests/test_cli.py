from __future__ import annotations

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.align.base import UnavailableWav2Vec2Backend
from whisper_omega.cli.main import main
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


class StereoTimedStubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None, word_timestamps=True):
        _ = (audio_path, model_name, language, device, batch_size, word_timestamps)
        return BackendTranscription(
            text="left right",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=0.45, text="left", speaker=None),
                Segment(id=1, start=0.55, end=1.0, text="right", speaker=None),
            ],
            words=[
                Word(text="left", start=0.0, end=0.45, speaker=None, confidence=0.9),
                Word(text="right", start=0.55, end=1.0, speaker=None, confidence=0.9),
            ],
        )


class NemoStubDiarizationBackend(DiarizationBackend):
    name = "nemo"

    def diarize(self, segments, words):
        speakers = [
            Speaker(id="SPEAKER_00", start=0.0, end=0.45, label="SPEAKER_00"),
            Speaker(id="SPEAKER_01", start=0.55, end=1.0, label="SPEAKER_01"),
        ]
        mapped_segments = [
            Segment(id=segments[0].id, start=segments[0].start, end=segments[0].end, text=segments[0].text, speaker="SPEAKER_00"),
            Segment(id=segments[1].id, start=segments[1].start, end=segments[1].end, text=segments[1].text, speaker="SPEAKER_01"),
        ]
        mapped_words = [
            Word(text=words[0].text, start=words[0].start, end=words[0].end, speaker="SPEAKER_00", confidence=words[0].confidence),
            Word(text=words[1].text, start=words[1].start, end=words[1].end, speaker="SPEAKER_01", confidence=words[1].confidence),
        ]
        return DiarizationOutcome(segments=mapped_segments, words=mapped_words, speakers=speakers)


class CustomStubDiarizationBackend(DiarizationBackend):
    name = "custom"

    def diarize(self, segments, words):
        speakers = [Speaker(id="speaker_0", start=0.0, end=1.0, label="speaker_0")]
        mapped_segments = [
            Segment(id=segment.id, start=segment.start, end=segment.end, text=segment.text, speaker="speaker_0")
            for segment in segments
        ]
        mapped_words = [
            Word(text=word.text, start=word.start, end=word.end, speaker="speaker_0", confidence=word.confidence)
            for word in words
        ]
        return DiarizationOutcome(segments=mapped_segments, words=mapped_words, speakers=speakers)


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
                    code="DIARIZATION_BACKEND_UNAVAILABLE",
                    category="dependency",
                    message="pyannote.audio is not installed",
                    retryable=False,
                )
            ],
        )


class TokenAwareDiarizationBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments, words):
        token = os.environ.get("HF_TOKEN")
        if not token:
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
        return DiarizationOutcome(segments=segments, words=words, speakers=[])


class HintAwareDiarizationBackend(DiarizationBackend):
    name = "pyannote"

    def diarize(self, segments, words):
        speakers = [
            Speaker(
                id=f"num={os.environ.get('OMEGA_PYANNOTE_NUM_SPEAKERS')},min={os.environ.get('OMEGA_PYANNOTE_MIN_SPEAKERS')},max={os.environ.get('OMEGA_PYANNOTE_MAX_SPEAKERS')}",
                start=0.0,
                end=1.0,
                label="hint",
            )
        ]
        return DiarizationOutcome(segments=segments, words=words, speakers=speakers)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmpdir.name) / "sample.wav"
        self.audio_path.write_bytes(b"RIFF")
        self.original_hf_token = os.environ.pop("HF_TOKEN", None)

    def tearDown(self) -> None:
        if self.original_hf_token is not None:
            os.environ["HF_TOKEN"] = self.original_hf_token
        else:
            os.environ.pop("HF_TOKEN", None)
        self.tmpdir.cleanup()

    def _write_stereo_wav(self, name: str) -> Path:
        path = Path(self.tmpdir.name) / name
        sample_rate = 8000
        frames = bytearray()
        for index in range(sample_rate):
            if index < sample_rate // 2:
                left, right = 20000, 500
            else:
                left, right = 500, 20000
            frames.extend(int(left).to_bytes(2, byteorder="little", signed=True))
            frames.extend(int(right).to_bytes(2, byteorder="little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frames))
        return path

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
                alignment_backend=UnavailableWav2Vec2Backend(),
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

    def test_emit_result_json_on_failure_only_suppresses_success_json(self) -> None:
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
                    "--emit-result-json",
                    "on-failure",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn('"schema_version"', result.output)
        self.assertIn("transcription completed", result.output)

    def test_emit_result_json_on_failure_emits_failure_json(self) -> None:
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
                    "--emit-result-json",
                    "on-failure",
                ],
            )

        self.assertEqual(result.exit_code, 30)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "failure")
        self.assertIn("failure:", result.output)

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
                diarization_backend=MissingDiarizationBackend(),
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

    def test_whisperx_align_model_requests_alignment(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    required_features=["alignment"],
                    align_backend="wav2vec2",
                ),
                asr_backend=StubBackend(),
                alignment_backend=UnavailableWav2Vec2Backend(),
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
                    "--align_model",
                    "ja",
                    "--output_format",
                    "json",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("transcription completed", result.output)
        self.assertIn("--align_model=ja requests alignment", result.output)

    def test_whisperx_hf_token_sets_environment_for_backend(self) -> None:
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
                diarization_backend=TokenAwareDiarizationBackend(),
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
                    "--hf_token",
                    "secret-token",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(os.environ.get("HF_TOKEN"), "secret-token")

    def test_whisperx_highlight_words_emits_partial_compatibility_warning(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
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
                    "--highlight_words",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--highlight_words is accepted for compatibility", result.output)

    def test_transcribe_accepts_channel_diarization_backend(self) -> None:
        stereo_path = self._write_stereo_wav("stereo.wav")
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    required_features=["diarization"],
                    diarize_backend="channel",
                ),
                asr_backend=StereoTimedStubBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(stereo_path),
                    "--device",
                    "cpu",
                    "--require-diarization",
                    "--diarize-backend",
                    "channel",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["metadata"]["diarization_backend"], "channel")
        self.assertEqual(payload["segments"][0]["speaker"], "CHANNEL_LEFT")
        self.assertEqual(payload["segments"][1]["speaker"], "CHANNEL_RIGHT")

    def test_transcribe_accepts_nemo_diarization_backend(self) -> None:
        stereo_path = self._write_stereo_wav("nemo.wav")
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    required_features=["diarization"],
                    diarize_backend="nemo",
                ),
                asr_backend=StereoTimedStubBackend(),
                diarization_backend=NemoStubDiarizationBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(stereo_path),
                    "--device",
                    "cpu",
                    "--require-diarization",
                    "--diarize-backend",
                    "nemo",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["metadata"]["diarization_backend"], "nemo")
        self.assertEqual(payload["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(payload["segments"][1]["speaker"], "SPEAKER_01")

    def test_transcribe_accepts_custom_diarization_backend(self) -> None:
        stereo_path = self._write_stereo_wav("custom.wav")
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    required_features=["diarization"],
                    diarize_backend="custom",
                ),
                asr_backend=StereoTimedStubBackend(),
                diarization_backend=CustomStubDiarizationBackend(),
            )
            service.transcribe.side_effect = real_service.transcribe
            service.write_output.side_effect = real_service.write_output
            service.config = real_service.config

            result = self.runner.invoke(
                main,
                [
                    "transcribe",
                    str(stereo_path),
                    "--device",
                    "cpu",
                    "--require-diarization",
                    "--diarize-backend",
                    "custom",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["metadata"]["diarization_backend"], "custom")
        self.assertEqual(payload["segments"][0]["speaker"], "speaker_0")

    def test_transcribe_passes_speaker_hint_options_to_diarization_backend(self) -> None:
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
                diarization_backend=HintAwareDiarizationBackend(),
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
                    "--require-diarization",
                    "--diarize-backend",
                    "pyannote",
                    "--num-speakers",
                    "2",
                    "--min-speakers",
                    "1",
                    "--max-speakers",
                    "3",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["speakers"][0]["id"], "num=2,min=1,max=3")

    def test_transcribe_rejects_inconsistent_speaker_hint_options(self) -> None:
        result = self.runner.invoke(
            main,
            [
                "transcribe",
                str(self.audio_path),
                "--require-diarization",
                "--diarize-backend",
                "pyannote",
                "--num-speakers",
                "2",
                "--min-speakers",
                "3",
            ],
        )

        self.assertEqual(result.exit_code, 40)
        self.assertIn("--num-speakers cannot be smaller than --min-speakers", result.output)

    def test_transcribe_can_disable_word_timestamps_and_word_output(self) -> None:
        captured = {}

        class WordTimestampAwareBackend(ASRBackend):
            name = "stub"

            def transcribe(self, audio_path, model_name, language, device, batch_size=None, word_timestamps=True):
                _ = (audio_path, model_name, language, device, batch_size)
                captured["word_timestamps"] = word_timestamps
                words = []
                if word_timestamps:
                    words.append(Word(text="hello", start=0.0, end=0.5, speaker=None, confidence=0.9))
                return BackendTranscription(
                    text="hello world",
                    language="en",
                    segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker=None)],
                    words=words,
                )

        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    word_timestamps=False,
                    include_words=False,
                ),
                asr_backend=WordTimestampAwareBackend(),
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
                    "--no-word-timestamps",
                    "--no-include-words",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertFalse(captured["word_timestamps"])
        payload = extract_json(result.output)
        self.assertEqual(payload["words"], [])
        self.assertEqual(payload["metadata"]["timestamp_strategy"], "segment_only")
        self.assertEqual(payload["metadata"]["timestamp_source"], "segments")
        self.assertEqual(payload["metadata"]["timestamp_quality"], "segment_only")

    def test_transcribe_accepts_explicit_timestamp_strategy(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    timestamp_strategy="direct",
                ),
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
                    "--timestamp-strategy",
                    "direct",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["metadata"]["requested_timestamp_strategy"], "direct")
        self.assertEqual(payload["metadata"]["timestamp_strategy"], "direct")

    def test_transcribe_rejects_timestamp_strategy_without_alignment_backend(self) -> None:
        result = self.runner.invoke(
            main,
            [
                "transcribe",
                str(self.audio_path),
                "--timestamp-strategy",
                "aligned",
                "--align-backend",
                "none",
            ],
        )

        self.assertEqual(result.exit_code, 40)
        self.assertIn("--timestamp-strategy aligned requires a non-none --align-backend", result.output)

    def test_transcribe_can_suppress_segment_output(self) -> None:
        with patch("whisper_omega.cli.main.TranscriptionService") as service_cls:
            service = service_cls.return_value
            from whisper_omega.runtime.service import ServiceConfig, TranscriptionService
            from whisper_omega.runtime.policy import PolicyConfig

            real_service = TranscriptionService(
                ServiceConfig(
                    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
                    include_segments=False,
                ),
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
                    "--no-include-segments",
                    "--emit-result-json",
                    "always",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        payload = extract_json(result.output)
        self.assertEqual(payload["segments"], [])

    def test_transcribe_rejects_segmentless_subtitle_output(self) -> None:
        result = self.runner.invoke(
            main,
            [
                "transcribe",
                str(self.audio_path),
                "--output-format",
                "srt",
                "--no-include-segments",
            ],
        )

        self.assertEqual(result.exit_code, 40)
        self.assertIn("requires segment output", result.output)

    def test_setup_align_lists_alignment_steps(self) -> None:
        result = self.runner.invoke(main, ["setup", "align"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Alignment setup path:", result.output)
        self.assertIn(".[align]", result.output)
        self.assertIn("OMEGA_ALIGNMENT_ROMANIZER", result.output)

    def test_setup_validation_lists_manifest_steps(self) -> None:
        result = self.runner.invoke(main, ["setup", "validation"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Validation setup path:", result.output)
        self.assertIn(".[validation]", result.output)
        self.assertIn("build_dataset_manifest.py", result.output)
        self.assertIn("export_google_fleurs_fixtures.py", result.output)
        self.assertIn("build_long_fixture.py", result.output)
        self.assertIn("build_diarization_fixture.py", result.output)
        self.assertIn("build_failure_fixtures.py", result.output)
        self.assertIn("generate_validation_report.py", result.output)

    def test_setup_diarize_lists_hf_token_step(self) -> None:
        result = self.runner.invoke(main, ["setup", "diarize"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Diarization setup path:", result.output)
        self.assertIn("--diarize-backend channel", result.output)
        self.assertIn("--diarize-backend nemo", result.output)
        self.assertIn("HF_TOKEN", result.output)
        self.assertIn("ffmpeg", result.output.lower())


if __name__ == "__main__":
    unittest.main()
