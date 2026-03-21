#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from whisper_omega.api import transcribe_file


def run_case(audio_path: Path, device: str, repeats: int, model_name: str) -> dict:
    timings: list[float] = []
    statuses: list[str] = []
    error_codes: list[str | None] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = transcribe_file(audio_path, device=device, runtime_policy="permissive", model_name=model_name)
        timings.append(time.perf_counter() - started)
        statuses.append(result.status)
        error_codes.append(result.error_code)
    return {
        "device": device,
        "repeats": repeats,
        "timings_seconds": [round(item, 4) for item in timings],
        "median_seconds": round(statistics.median(timings), 4),
        "mean_seconds": round(statistics.fmean(timings), 4),
        "statuses": statuses,
        "error_codes": error_codes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight local benchmark for whisper-omega.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    args = parser.parse_args()

    payload = {
        "audio_path": str(args.audio_path),
        "model": args.model,
        "cases": [run_case(args.audio_path, device=device, repeats=args.repeats, model_name=args.model) for device in args.devices],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
