#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whisper_omega import transcribe_file
from whisper_omega.runtime.service import DoctorReport


def _write_smoke_wav(path: Path) -> None:
    framerate = 16000
    samples = []
    for i in range(framerate):
        value = int(8000 * math.sin(2 * math.pi * 440 * i / framerate)) if i < framerate // 4 else 0
        samples.append(value)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _case_payload(name: str, expectation: str, result=None, blocked_reason: str | None = None) -> dict:
    payload = {
        "name": name,
        "expectation": expectation,
        "blocked_reason": blocked_reason,
        "status": None,
        "error_code": None,
        "error_category": None,
        "speaker_count": None,
        "passed": False,
    }
    if result is not None:
        payload.update(
            {
                "status": result.status,
                "error_code": result.error_code,
                "error_category": result.error_category,
                "speaker_count": len(result.speakers),
            }
        )
    return payload


def build_pyannote_acceptance_report() -> dict:
    doctor = DoctorReport.collect().to_dict()
    pyannote_status = doctor["backend_statuses"]["diarization"]["pyannote"]
    decode_status = doctor["backend_statuses"]["decode"]
    blocked_reasons: list[str] = []
    if not doctor["faster_whisper_available"]:
        blocked_reasons.append("DEPENDENCY_MISSING")
    if not pyannote_status["installed"]:
        blocked_reasons.append("DIARIZATION_BACKEND_UNAVAILABLE")
    if not (
        decode_status["torchaudio"]["ready"]
        or decode_status["ffmpeg_torchcodec"]["ready"]
    ):
        blocked_reasons.append("DIARIZATION_DECODE_UNAVAILABLE")

    with tempfile.TemporaryDirectory(prefix="omega-pyannote-acceptance-") as tmpdir:
        audio_path = Path(tmpdir) / "pyannote_acceptance.wav"
        _write_smoke_wav(audio_path)

        cases: list[dict] = []
        base_kwargs = {
            "audio_path": audio_path,
            "device": "cpu",
            "model_name": "tiny",
            "required_features": ["diarization"],
            "diarize_backend": "pyannote",
        }

        if blocked_reasons:
            cases.append(
                {
                    **_case_payload(
                        "pyannote_missing_token",
                        "degraded_or_failure_with_HF_TOKEN_MISSING",
                        blocked_reason=",".join(blocked_reasons),
                    ),
                    "passed": False,
                }
            )
            cases.append(
                {
                    **_case_payload(
                        "pyannote_with_token_and_hints",
                        "success_with_speaker_hints",
                        blocked_reason=",".join(blocked_reasons),
                    ),
                    "passed": False,
                }
            )
            return {
                "report_version": "0.1",
                "doctor": doctor,
                "cases": cases,
                "blocked": True,
                "blocked_reasons": blocked_reasons,
                "all_passed": False,
            }

        with _temporary_env(
            {
                "HF_TOKEN": None,
                "OMEGA_PYANNOTE_NUM_SPEAKERS": None,
                "OMEGA_PYANNOTE_MIN_SPEAKERS": None,
                "OMEGA_PYANNOTE_MAX_SPEAKERS": None,
            }
        ):
            missing_token_result = transcribe_file(**base_kwargs)
        missing_token_case = _case_payload(
            "pyannote_missing_token",
            "degraded_or_failure_with_HF_TOKEN_MISSING",
            result=missing_token_result,
        )
        missing_token_case["passed"] = missing_token_result.error_code == "HF_TOKEN_MISSING"
        cases.append(missing_token_case)

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            cases.append(
                {
                    **_case_payload(
                        "pyannote_with_token_and_hints",
                        "success_with_speaker_hints",
                        blocked_reason="HF_TOKEN_MISSING",
                    ),
                    "passed": False,
                }
            )
            return {
                "report_version": "0.1",
                "doctor": doctor,
                "cases": cases,
                "blocked": True,
                "blocked_reasons": ["HF_TOKEN_MISSING"],
                "all_passed": all(case["passed"] for case in cases),
            }

        with _temporary_env(
            {
                "HF_TOKEN": hf_token,
                "OMEGA_PYANNOTE_MIN_SPEAKERS": "1",
                "OMEGA_PYANNOTE_MAX_SPEAKERS": "2",
            }
        ):
            token_result = transcribe_file(**base_kwargs)
        token_case = _case_payload(
            "pyannote_with_token_and_hints",
            "success_with_speaker_hints",
            result=token_result,
        )
        token_case["passed"] = token_result.status == "success"
        cases.append(token_case)

        return {
            "report_version": "0.1",
            "doctor": doctor,
            "cases": cases,
            "blocked": False,
            "blocked_reasons": [],
            "all_passed": all(case["passed"] for case in cases),
        }


def main() -> int:
    report = build_pyannote_acceptance_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blocked"]:
        return 2
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
