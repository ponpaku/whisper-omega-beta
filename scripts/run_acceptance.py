#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_validation_report import build_report


DEFAULT_OUTPUT = ROOT / "validation-report.json"


def acceptance_failures(report: dict) -> list[str]:
    failures: list[str] = []

    if report["doctor"]["returncode"] != 0:
        failures.append("doctor")
    if report["tests"]["returncode"] != 0:
        failures.append("tests")

    for key, label in (
        ("smoke", "asr_smoke"),
        ("alignment_smoke", "alignment_smoke"),
        ("diarization_smoke", "diarization_smoke"),
    ):
        section = report.get(key)
        if not section:
            continue
        if section["returncode"] != 0:
            failures.append(label)
            continue
        payload = section.get("payload")
        if isinstance(payload, dict) and payload.get("all_passed") is False:
            failures.append(label)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical local acceptance flow.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the acceptance report JSON.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the ASR smoke step.",
    )
    parser.add_argument(
        "--skip-alignment-smoke",
        action="store_true",
        help="Skip the alignment smoke step.",
    )
    parser.add_argument(
        "--skip-diarization-smoke",
        action="store_true",
        help="Skip the diarization smoke step.",
    )
    args = parser.parse_args()

    report = build_report(
        include_smoke=not args.skip_smoke,
        include_alignment_smoke=not args.skip_alignment_smoke,
        include_diarization_smoke=not args.skip_diarization_smoke,
    )

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = acceptance_failures(report)
    print(f"Acceptance report written to {args.output}")
    print(f"doctor: {report['doctor']['returncode']}")
    print(f"tests: {report['tests']['returncode']}")
    for key, label in (
        ("smoke", "asr_smoke"),
        ("alignment_smoke", "alignment_smoke"),
        ("diarization_smoke", "diarization_smoke"),
    ):
        section = report.get(key)
        if section is None:
            print(f"{label}: skipped")
        else:
            payload = section.get("payload")
            if isinstance(payload, dict) and "all_passed" in payload:
                suffix = f", all_passed={payload['all_passed']}"
            else:
                suffix = ""
            print(f"{label}: {section['returncode']}{suffix}")

    if failures:
        print(f"acceptance: failed ({', '.join(failures)})", file=sys.stderr)
        return 1

    print("acceptance: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
