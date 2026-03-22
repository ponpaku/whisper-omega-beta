#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_report(
    include_smoke: bool,
    include_alignment_smoke: bool = False,
    include_diarization_smoke: bool = False,
    include_pyannote_acceptance: bool = False,
    include_nemo_acceptance: bool = False,
    include_gpu_acceptance: bool = False,
) -> dict:
    python_exe = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    doctor = subprocess.run(
        [python_exe, "-m", "whisper_omega", "doctor", "--json-output"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    doctor_payload = None
    if doctor.returncode == 0:
        try:
            doctor_payload = json.loads(doctor.stdout)
        except json.JSONDecodeError:
            doctor_payload = None

    tests = subprocess.run(
        [python_exe, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    report = {
        "report_version": "0.1",
        "workspace": str(ROOT),
        "python_executable": python_exe,
        "platform": platform.platform(),
        "doctor": {
            "returncode": doctor.returncode,
            "payload": doctor_payload,
            "stdout": doctor.stdout,
            "stderr": doctor.stderr,
        },
        "tests": {
            "returncode": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        },
        "smoke": None,
        "alignment_smoke": None,
        "diarization_smoke": None,
        "pyannote_acceptance": None,
        "nemo_acceptance": None,
        "gpu_acceptance": None,
    }

    if include_smoke:
        smoke = subprocess.run(
            ["/bin/bash", "scripts/run_smoke.sh"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        report["smoke"] = {
            "returncode": smoke.returncode,
            "stdout": smoke.stdout,
            "stderr": smoke.stderr,
        }

    if include_alignment_smoke:
        alignment_smoke = subprocess.run(
            [python_exe, "scripts/run_alignment_smoke.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        alignment_smoke_payload = None
        if alignment_smoke.returncode == 0:
            try:
                alignment_smoke_payload = json.loads(alignment_smoke.stdout)
            except json.JSONDecodeError:
                alignment_smoke_payload = None
        report["alignment_smoke"] = {
            "returncode": alignment_smoke.returncode,
            "payload": alignment_smoke_payload,
            "stdout": alignment_smoke.stdout,
            "stderr": alignment_smoke.stderr,
        }

    if include_diarization_smoke:
        diarization_smoke = subprocess.run(
            [python_exe, "scripts/run_diarization_smoke.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        diarization_smoke_payload = None
        if diarization_smoke.returncode == 0:
            try:
                diarization_smoke_payload = json.loads(diarization_smoke.stdout)
            except json.JSONDecodeError:
                diarization_smoke_payload = None
        report["diarization_smoke"] = {
            "returncode": diarization_smoke.returncode,
            "payload": diarization_smoke_payload,
            "stdout": diarization_smoke.stdout,
            "stderr": diarization_smoke.stderr,
        }

    if include_pyannote_acceptance:
        pyannote_acceptance = subprocess.run(
            [python_exe, "scripts/run_pyannote_acceptance.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        pyannote_acceptance_payload = None
        if pyannote_acceptance.stdout:
            try:
                pyannote_acceptance_payload = json.loads(pyannote_acceptance.stdout)
            except json.JSONDecodeError:
                pyannote_acceptance_payload = None
        report["pyannote_acceptance"] = {
            "returncode": pyannote_acceptance.returncode,
            "payload": pyannote_acceptance_payload,
            "stdout": pyannote_acceptance.stdout,
            "stderr": pyannote_acceptance.stderr,
        }

    if include_nemo_acceptance:
        nemo_acceptance = subprocess.run(
            [python_exe, "scripts/run_nemo_acceptance.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        nemo_acceptance_payload = None
        if nemo_acceptance.stdout:
            try:
                nemo_acceptance_payload = json.loads(nemo_acceptance.stdout)
            except json.JSONDecodeError:
                nemo_acceptance_payload = None
        report["nemo_acceptance"] = {
            "returncode": nemo_acceptance.returncode,
            "payload": nemo_acceptance_payload,
            "stdout": nemo_acceptance.stdout,
            "stderr": nemo_acceptance.stderr,
        }

    if include_gpu_acceptance:
        gpu_acceptance = subprocess.run(
            [python_exe, "scripts/run_gpu_acceptance.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        gpu_acceptance_payload = None
        if gpu_acceptance.stdout:
            try:
                gpu_acceptance_payload = json.loads(gpu_acceptance.stdout)
            except json.JSONDecodeError:
                gpu_acceptance_payload = None
        report["gpu_acceptance"] = {
            "returncode": gpu_acceptance.returncode,
            "payload": gpu_acceptance_payload,
            "stdout": gpu_acceptance.stdout,
            "stderr": gpu_acceptance.stderr,
        }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local validation report JSON.")
    parser.add_argument("--include-smoke", action="store_true", help="Run scripts/run_smoke.sh as part of the report.")
    parser.add_argument(
        "--include-alignment-smoke",
        action="store_true",
        help="Run scripts/run_alignment_smoke.py as part of the report.",
    )
    parser.add_argument(
        "--include-diarization-smoke",
        action="store_true",
        help="Run scripts/run_diarization_smoke.py as part of the report.",
    )
    parser.add_argument(
        "--include-pyannote-acceptance",
        action="store_true",
        help="Run scripts/run_pyannote_acceptance.py as part of the report.",
    )
    parser.add_argument(
        "--include-nemo-acceptance",
        action="store_true",
        help="Run scripts/run_nemo_acceptance.py as part of the report.",
    )
    parser.add_argument(
        "--include-gpu-acceptance",
        action="store_true",
        help="Run scripts/run_gpu_acceptance.py as part of the report.",
    )
    parser.add_argument("--output", type=Path, help="Optional output path for the report JSON.")
    args = parser.parse_args()

    report = build_report(
        include_smoke=args.include_smoke,
        include_alignment_smoke=args.include_alignment_smoke,
        include_diarization_smoke=args.include_diarization_smoke,
        include_pyannote_acceptance=args.include_pyannote_acceptance,
        include_nemo_acceptance=args.include_nemo_acceptance,
        include_gpu_acceptance=args.include_gpu_acceptance,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
