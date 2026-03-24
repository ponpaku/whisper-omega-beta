from __future__ import annotations

import unittest

from whisper_omega.runtime.models import BackendError, Metadata, Segment, Speaker, Timings, TranscriptionResult, Word


class ModelTests(unittest.TestCase):
    def test_success_requires_no_errors(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptionResult(
                schema_version="1.0.0",
                status="success",
                text="hello",
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="hello", speaker=None)],
                words=[Word(text="hello", start=0.0, end=1.0, speaker=None, confidence=0.12345)],
                speakers=[],
                metadata=Metadata(
                    asr_backend="stub",
                    align_backend="none",
                    diarization_backend="none",
                    device="cpu",
                    requested_device="cpu",
                    actual_device="cpu",
                ),
                backend_errors=[
                    BackendError(
                        backend="stub",
                        code="ERR",
                        category="runtime",
                        message="bad",
                        retryable=False,
                    )
                ],
            )

    def test_rounding_contract(self) -> None:
        result = TranscriptionResult(
            schema_version="1.0.0",
            status="failure",
            text="",
            language="",
            segments=[],
            words=[Word(text="hello", start=0.12349, end=1.99994, speaker=None, confidence=0.123456)],
            speakers=[Speaker(id="SPEAKER_00", start=0.12349, end=1.99994, label="Speaker 0")],
            metadata=Metadata(
                asr_backend="stub",
                align_backend="none",
                diarization_backend="none",
                device="cpu",
                requested_device="cpu",
                actual_device="cpu",
            ),
            error_code="DEPENDENCY_MISSING",
            error_category="dependency",
        )
        word = result.to_dict()["words"][0]
        speaker = result.to_dict()["speakers"][0]
        self.assertEqual(word["start"], 0.123)
        self.assertEqual(word["end"], 2.0)
        self.assertEqual(word["confidence"], 0.1235)
        self.assertEqual(speaker["start"], 0.123)
        self.assertEqual(speaker["end"], 2.0)

    def test_segment_speaker_must_exist_in_speakers(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptionResult(
                schema_version="1.0.0",
                status="success",
                text="hello",
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="hello", speaker="SPEAKER_99")],
                words=[Word(text="hello", start=0.0, end=1.0, speaker=None, confidence=0.9)],
                speakers=[Speaker(id="SPEAKER_00", start=0.0, end=1.0, label="Speaker 0")],
                metadata=Metadata(
                    asr_backend="stub",
                    align_backend="none",
                    diarization_backend="pyannote",
                    device="cpu",
                    requested_device="cpu",
                    actual_device="cpu",
                ),
            )

    def test_word_speaker_must_exist_in_speakers(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptionResult(
                schema_version="1.0.0",
                status="success",
                text="hello",
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="hello", speaker="SPEAKER_00")],
                words=[Word(text="hello", start=0.0, end=1.0, speaker="SPEAKER_99", confidence=0.9)],
                speakers=[Speaker(id="SPEAKER_00", start=0.0, end=1.0, label="Speaker 0")],
                metadata=Metadata(
                    asr_backend="stub",
                    align_backend="none",
                    diarization_backend="pyannote",
                    device="cpu",
                    requested_device="cpu",
                    actual_device="cpu",
                ),
            )

    def test_duplicate_speaker_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptionResult(
                schema_version="1.0.0",
                status="failure",
                text="",
                language="",
                segments=[],
                words=[],
                speakers=[
                    Speaker(id="SPEAKER_00", start=0.0, end=1.0, label="Speaker 0"),
                    Speaker(id="SPEAKER_00", start=1.0, end=2.0, label="Speaker 0 duplicate"),
                ],
                metadata=Metadata(
                    asr_backend="stub",
                    align_backend="none",
                    diarization_backend="pyannote",
                    device="cpu",
                    requested_device="cpu",
                    actual_device="cpu",
                ),
                error_code="GPU_UNAVAILABLE",
                error_category="runtime",
            )

    def test_metadata_serializes_timings(self) -> None:
        metadata = Metadata(
            asr_backend="stub",
            align_backend="none",
            diarization_backend="none",
            device="cpu",
            requested_device="cpu",
            actual_device="cpu",
            timings=Timings(
                total_ms=100,
                asr_ms=60,
                alignment_ms=20,
                diarization_ms=10,
                audio_duration_ms=200,
                real_time_factor=0.5,
            ),
        )

        self.assertEqual(
            metadata.to_dict()["timings"],
            {
                "total_ms": 100,
                "asr_ms": 60,
                "alignment_ms": 20,
                "diarization_ms": 10,
                "audio_duration_ms": 200,
                "real_time_factor": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
