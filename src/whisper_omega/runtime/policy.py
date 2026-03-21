from __future__ import annotations

from dataclasses import dataclass


class UsageError(ValueError):
    """Raised when CLI arguments violate the contract."""


@dataclass(slots=True)
class PolicyConfig:
    runtime_policy: str
    device: str


def validate_cli_constraints(
    runtime_policy: str,
    device: str,
    output_format: str,
    output_file: str | None,
    write_failure_json: bool,
    diarize_backend: str,
    align_backend: str,
    required_features: tuple[str, ...],
    require_alignment: bool,
    require_diarization: bool,
) -> list[str]:
    normalized = list(required_features)
    if require_alignment and "alignment" not in normalized:
        normalized.append("alignment")
    if require_diarization and "diarization" not in normalized:
        normalized.append("diarization")

    if runtime_policy == "strict-gpu" and device == "cpu":
        raise UsageError("--runtime-policy strict-gpu cannot be combined with --device cpu")
    if output_format == "txt" and write_failure_json and not output_file:
        raise UsageError("--output-format txt with --write-failure-json requires --output-file")
    if diarize_backend == "none" and "diarization" in normalized:
        raise UsageError("--diarize-backend none cannot satisfy diarization requirement")
    if align_backend == "none" and "alignment" in normalized:
        raise UsageError("--align-backend none cannot satisfy alignment requirement")
    return normalized


def effective_device(device: str) -> str:
    if device == "auto":
        return "cpu"
    return device

