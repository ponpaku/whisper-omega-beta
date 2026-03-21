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


def build_report(include_smoke: bool) -> dict:
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

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local validation report JSON.")
    parser.add_argument("--include-smoke", action="store_true", help="Run scripts/run_smoke.sh as part of the report.")
    parser.add_argument("--output", type=Path, help="Optional output path for the report JSON.")
    args = parser.parse_args()

    report = build_report(include_smoke=args.include_smoke)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
