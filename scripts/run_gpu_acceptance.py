#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whisper_omega import transcribe_file
from whisper_omega.runtime.policy import cuda_available
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


def _case_payload(name: str, expectation: str, result=None, blocked_reason: str | None = None) -> dict:
    payload = {
        "name": name,
        "expectation": expectation,
        "blocked_reason": blocked_reason,
        "status": None,
        "error_code": None,
        "error_category": None,
        "requested_device": None,
        "actual_device": None,
        "runtime_policy": None,
        "passed": False,
        "residual_risk": None,
    }
    if result is not None:
        payload.update(
            {
                "status": result.status,
                "error_code": result.error_code,
                "error_category": result.error_category,
                "requested_device": result.metadata.requested_device,
                "actual_device": result.metadata.actual_device,
                "runtime_policy": None,
            }
        )
    return payload


def _passed_gpu_selection_case(result, expected_requested_device: str) -> tuple[bool, str | None]:
    if result.metadata.requested_device != expected_requested_device:
        return (False, None)
    if result.metadata.actual_device != "cuda":
        return (False, None)
    if result.error_code == "GPU_UNAVAILABLE":
        return (False, None)
    if result.error_code == "AUDIO_DECODE_FAILURE":
        return (True, "AUDIO_DECODE_FAILURE")
    return (True, None)


def build_gpu_acceptance_report() -> dict:
    doctor = DoctorReport.collect().to_dict()
    blocked_reasons: list[str] = []
    if not doctor["faster_whisper_available"]:
        blocked_reasons.append("DEPENDENCY_MISSING")
    if not doctor["torch_cuda_available"] or not cuda_available():
        blocked_reasons.append("GPU_UNAVAILABLE")

    with tempfile.TemporaryDirectory(prefix="omega-gpu-acceptance-") as tmpdir:
        audio_path = Path(tmpdir) / "gpu_acceptance.wav"
        _write_smoke_wav(audio_path)
        cases: list[dict] = []

        if blocked_reasons:
            for name, expectation in (
                ("gpu_auto_prefers_cuda", "actual_device=cuda without silent cpu fallback"),
                ("gpu_cuda_requires_gpu", "requested_device=cuda and actual_device=cuda"),
                ("gpu_strict_gpu_enforces_cuda", "strict-gpu must not silently fall back to cpu"),
            ):
                cases.append(
                    {
                        **_case_payload(name, expectation, blocked_reason=",".join(blocked_reasons)),
                        "passed": False,
                    }
                )
            return {
                "report_version": "0.1",
                "doctor": doctor,
                "cases": cases,
                "blocked": True,
                "blocked_reasons": blocked_reasons,
                "residual_risks": [],
                "all_passed": False,
            }

        run_specs = [
            ("gpu_auto_prefers_cuda", "actual_device=cuda without silent cpu fallback", {"device": "auto", "runtime_policy": "permissive"}),
            ("gpu_cuda_requires_gpu", "requested_device=cuda and actual_device=cuda", {"device": "cuda", "runtime_policy": "permissive"}),
            ("gpu_strict_gpu_enforces_cuda", "strict-gpu must not silently fall back to cpu", {"device": "auto", "runtime_policy": "strict-gpu"}),
        ]
        residual_risks: list[str] = []

        for name, expectation, config in run_specs:
            result = transcribe_file(
                audio_path,
                device=config["device"],
                runtime_policy=config["runtime_policy"],
                model_name="tiny",
            )
            case = _case_payload(name, expectation, result=result)
            case["runtime_policy"] = config["runtime_policy"]
            expected_requested_device = config["device"]
            if config["device"] == "auto":
                expected_requested_device = "auto"
            passed, residual = _passed_gpu_selection_case(result, expected_requested_device)
            case["passed"] = passed
            case["residual_risk"] = residual
            if residual:
                residual_risks.append(f"{name}:{residual}")
            cases.append(case)

        return {
            "report_version": "0.1",
            "doctor": doctor,
            "cases": cases,
            "blocked": False,
            "blocked_reasons": [],
            "residual_risks": residual_risks,
            "all_passed": all(case["passed"] for case in cases),
        }


def main() -> int:
    report = build_gpu_acceptance_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blocked"]:
        return 2
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
