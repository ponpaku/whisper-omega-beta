from __future__ import annotations

import unittest

from whisper_omega.runtime.models import BackendError, Metadata, Segment, Speaker, TranscriptionResult, Word


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


if __name__ == "__main__":
    unittest.main()
