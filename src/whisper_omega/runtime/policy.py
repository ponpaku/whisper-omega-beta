from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import warnings

TIMESTAMP_STRATEGY_CHOICES = (
    "segment_only",
    "direct",
    "aligned",
    "hybrid_selective_align",
)


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


def resolve_timestamp_strategy(
    timestamp_strategy: str | None,
    *,
    word_timestamps: bool,
    align_backend: str,
    required_features: list[str] | tuple[str, ...],
) -> str:
    if timestamp_strategy is None:
        if not word_timestamps:
            return "segment_only"
        if "alignment" in required_features and align_backend != "none":
            return "aligned"
        return "direct"

    if timestamp_strategy not in TIMESTAMP_STRATEGY_CHOICES:
        raise UsageError(
            f"--timestamp-strategy must be one of: {', '.join(TIMESTAMP_STRATEGY_CHOICES)}"
        )
    if timestamp_strategy == "segment_only" and word_timestamps:
        raise UsageError("--timestamp-strategy segment_only requires --no-word-timestamps")
    if timestamp_strategy in {"direct", "aligned", "hybrid_selective_align"} and not word_timestamps:
        raise UsageError(
            f"--timestamp-strategy {timestamp_strategy} cannot be combined with --no-word-timestamps"
        )
    if timestamp_strategy in {"aligned", "hybrid_selective_align"} and align_backend == "none":
        raise UsageError(
            f"--timestamp-strategy {timestamp_strategy} requires a non-none --align-backend"
        )
    return timestamp_strategy


def effective_device(device: str) -> str:
    if device != "auto":
        return device
    if cuda_available():
        return "cuda"
    return "cpu"


def cuda_available() -> bool:
    try:
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if torch.cuda.is_available():
                return True
    except Exception:
        pass

    try:
        if shutil.which("nvidia-smi") is None:
            return False
    except Exception:
        return False

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return False

    return bool(completed.stdout.strip())
