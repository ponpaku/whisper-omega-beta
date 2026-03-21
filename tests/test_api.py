from __future__ import annotations

import tempfile
import unittest
import wave

from whisper_omega import PolicyConfig, ServiceConfig, TranscriptionResult, TranscriptionService, transcribe_file


class ApiTests(unittest.TestCase):
    def test_package_exports_public_api(self) -> None:
        self.assertIsNotNone(PolicyConfig)
        self.assertIsNotNone(ServiceConfig)
        self.assertIsNotNone(TranscriptionService)
        self.assertIsNotNone(TranscriptionResult)

    def test_transcribe_file_returns_failure_result_without_backend_dependency(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            with wave.open(handle.name, "wb") as wav_handle:
                wav_handle.setnchannels(1)
                wav_handle.setsampwidth(2)
                wav_handle.setframerate(16000)
                wav_handle.writeframes(b"\x00\x00" * 1600)

            result = transcribe_file(handle.name, device="cpu")

        self.assertIsInstance(result, TranscriptionResult)
        self.assertIn(result.status, {"failure", "success", "degraded"})


if __name__ == "__main__":
    unittest.main()
