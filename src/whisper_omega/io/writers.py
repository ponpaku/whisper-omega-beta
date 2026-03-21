from __future__ import annotations

import json
from pathlib import Path

from whisper_omega.runtime.models import TranscriptionResult


def write_json(result: TranscriptionResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_txt(result: TranscriptionResult, path: Path) -> None:
    path.write_text(result.text + "\n", encoding="utf-8")


def _subtitle_timestamp(seconds: float, separator: str) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def write_srt(result: TranscriptionResult, path: Path) -> None:
    blocks: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_subtitle_timestamp(segment.start, ',')} --> {_subtitle_timestamp(segment.end, ',')}",
                    segment.text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def write_vtt(result: TranscriptionResult, path: Path) -> None:
    blocks = ["WEBVTT"]
    for segment in result.segments:
        blocks.append(
            "\n".join(
                [
                    "",
                    f"{_subtitle_timestamp(segment.start, '.')} --> {_subtitle_timestamp(segment.end, '.')}",
                    segment.text,
                ]
            )
        )
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")

