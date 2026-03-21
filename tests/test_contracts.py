from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from whisper_omega.asr.base import ASRBackend, BackendTranscription
from whisper_omega.runtime.models import Metadata, Segment, Speaker, TranscriptionResult, Word
from whisper_omega.runtime.policy import effective_device
from whisper_omega.runtime.service import DoctorReport


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "whisper-omega-plan" / "specs" / "appendix_a_schema_v01.json"


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio_path, model_name, language, device, batch_size=None):
        _ = (audio_path, model_name, language, device, batch_size)
        return BackendTranscription(
            text="hello world",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker="SPEAKER_00")],
            words=[Word(text="hello", start=0.0, end=0.5, speaker="SPEAKER_00", confidence=0.9)],
            backend_errors=[],
        )


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(schema)

    def test_result_schema_accepts_success_payload(self) -> None:
        result = TranscriptionResult(
            schema_version="1.0.0",
            status="success",
            text="hello world",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", speaker="SPEAKER_00")],
            words=[Word(text="hello", start=0.0, end=0.5, speaker="SPEAKER_00", confidence=0.9)],
            speakers=[Speaker(id="SPEAKER_00", start=0.0, end=1.0, label="Speaker 0")],
            metadata=Metadata(
                asr_backend="stub",
                align_backend="none",
                diarization_backend="pyannote",
                device="cpu",
                requested_device="auto",
                actual_device="cpu",
                requested_features=["asr", "diarization"],
                completed_features=["asr", "diarization"],
                failed_features=[],
            ),
        )

        self.validator.validate(result.to_dict())

    def test_result_schema_accepts_failure_payload(self) -> None:
        result = TranscriptionResult(
            schema_version="1.0.0",
            status="failure",
            text="",
            language="",
            segments=[],
            words=[],
            speakers=[],
            metadata=Metadata(
                asr_backend="stub",
                align_backend="wav2vec2",
                diarization_backend="none",
                device="cpu",
                requested_device="cuda",
                actual_device="cpu",
                requested_features=["asr", "alignment"],
                completed_features=[],
                failed_features=["asr"],
            ),
            error_code="GPU_UNAVAILABLE",
            error_category="runtime",
        )

        self.validator.validate(result.to_dict())

    def test_effective_device_prefers_cuda_when_available(self) -> None:
        with patch("whisper_omega.runtime.policy.cuda_available", return_value=True):
            self.assertEqual(effective_device("auto"), "cuda")

    def test_doctor_report_includes_extended_fields(self) -> None:
        report = DoctorReport.collect().to_dict()

        for key in (
            "ctranslate2_available",
            "torch_available",
            "torchaudio_available",
            "torchcodec_available",
            "torchcodec_importable",
            "torch_cuda_available",
            "ffmpeg_available",
            "hf_token_configured",
            "diarization_backend_available",
            "alignment_backend_available",
            "diarization_ready",
            "diarization_issue_code",
            "diarization_decode_ready",
            "diarization_decode_backend",
            "alignment_ready",
            "alignment_issue_code",
            "alignment_romanizer_configured",
            "alignment_language_strategy",
            "cache_dir",
            "cache_dir_writable",
            "detected_device",
            "known_issue_codes",
            "recommended_actions",
        ):
            self.assertIn(key, report)

    def test_doctor_prefers_torchaudio_for_diarization_decode(self) -> None:
        availability = {
            "faster_whisper": True,
            "ctranslate2": True,
            "torch": True,
            "torchaudio": True,
            "torchcodec": False,
            "pyannote.audio": True,
        }

        with patch("whisper_omega.runtime.service._module_available", side_effect=lambda name: availability.get(name, False)):
            with patch("whisper_omega.runtime.service._module_importable", return_value=False):
                with patch("whisper_omega.runtime.service.UnavailablePyannoteBackend.capability", return_value=(True, None)):
                    with patch("whisper_omega.runtime.service.Wav2Vec2AlignmentBackend.capability", return_value=(True, None)):
                        with patch("whisper_omega.runtime.service.shutil.which", return_value=None):
                            report = DoctorReport.collect().to_dict()

        self.assertTrue(report["diarization_decode_ready"])
        self.assertEqual(report["diarization_decode_backend"], "torchaudio")


if __name__ == "__main__":
    unittest.main()
