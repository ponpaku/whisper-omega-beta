from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from whisper_omega.compat.whisperx import map_whisperx_options
from whisper_omega.runtime.codes import EXIT_CODES
from whisper_omega.runtime.models import ResultStatus
from whisper_omega.runtime.policy import PolicyConfig, UsageError, validate_cli_constraints
from whisper_omega.runtime.service import DoctorReport, ServiceConfig, TranscriptionService


def _emit_result_json(mode: str, status: ResultStatus, stdout_is_tty: bool) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "on-failure":
        return status == "failure"
    return not stdout_is_tty


def _serialize(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def _write_output(service: TranscriptionService, output_path: Path | None, output_format: str, result) -> None:
    if output_path is None:
        return
    if result.status == "failure" and not service.config.write_failure_json:
        return
    service.write_output(result, output_format, output_path)


def _run_transcribe(
    input_path: str,
    model: str,
    language: str | None,
    device: str,
    runtime_policy: str,
    required_features: tuple[str, ...],
    require_alignment: bool,
    require_diarization: bool,
    output_format: str,
    output_file: str | None,
    emit_result_json: str,
    write_failure_json: bool,
    diarize_backend: str,
    align_backend: str,
    batch_size: int | None,
) -> int:
    try:
        normalized_required = validate_cli_constraints(
            runtime_policy=runtime_policy,
            device=device,
            output_format=output_format,
            output_file=output_file,
            write_failure_json=write_failure_json,
            diarize_backend=diarize_backend,
            align_backend=align_backend,
            required_features=required_features,
            require_alignment=require_alignment,
            require_diarization=require_diarization,
        )
    except UsageError as exc:
        click.echo(f"usage error: {exc}", err=True)
        return EXIT_CODES["usage"]

    config = ServiceConfig(
        policy=PolicyConfig(runtime_policy=runtime_policy, device=device),
        required_features=normalized_required,
        output_format=output_format,
        emit_result_json=emit_result_json,
        write_failure_json=write_failure_json,
        diarize_backend=diarize_backend,
        align_backend=align_backend,
        model_name=model,
        language=language,
        batch_size=batch_size,
    )
    service = TranscriptionService(config)
    result = service.transcribe(Path(input_path))

    if output_file:
        _write_output(service, Path(output_file), output_format, result)

    if _emit_result_json(emit_result_json, result.status, sys.stdout.isatty()):
        click.echo(_serialize(result.to_dict()))

    if result.status == "success":
        click.echo("transcription completed", err=True)
    elif result.status == "degraded":
        click.echo(f"degraded: {result.error_code or 'partial capability loss'}", err=True)
    else:
        click.echo(f"failure: {result.error_code or 'unknown'}", err=True)

    return EXIT_CODES[result.exit_family]


@click.group()
def main() -> None:
    """whisper-omega CLI."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--model", default="small", show_default=True)
@click.option("--language")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option(
    "--runtime-policy",
    type=click.Choice(["permissive", "strict", "strict-gpu"]),
    default="permissive",
    show_default=True,
)
@click.option("--require-feature", "required_features", multiple=True)
@click.option("--require-alignment", is_flag=True, default=False)
@click.option("--require-diarization", is_flag=True, default=False)
@click.option("--output-format", type=click.Choice(["json", "srt", "vtt", "txt"]), default="json", show_default=True)
@click.option("--output-file")
@click.option(
    "--emit-result-json",
    type=click.Choice(["auto", "always", "on-failure", "never"]),
    default="auto",
    show_default=True,
)
@click.option("--write-failure-json", is_flag=True, default=False)
@click.option("--diarize-backend", type=click.Choice(["none", "pyannote"]), default="none", show_default=True)
@click.option("--align-backend", type=click.Choice(["none", "wav2vec2"]), default="none", show_default=True)
@click.option("--batch-size", type=int)
def transcribe(
    input_path: str,
    model: str,
    language: str | None,
    device: str,
    runtime_policy: str,
    required_features: tuple[str, ...],
    require_alignment: bool,
    require_diarization: bool,
    output_format: str,
    output_file: str | None,
    emit_result_json: str,
    write_failure_json: bool,
    diarize_backend: str,
    align_backend: str,
    batch_size: int | None,
) -> None:
    """Transcribe an input audio file."""
    raise SystemExit(
        _run_transcribe(
            input_path=input_path,
            model=model,
            language=language,
            device=device,
            runtime_policy=runtime_policy,
            required_features=required_features,
            require_alignment=require_alignment,
            require_diarization=require_diarization,
            output_format=output_format,
            output_file=output_file,
            emit_result_json=emit_result_json,
            write_failure_json=write_failure_json,
            diarize_backend=diarize_backend,
            align_backend=align_backend,
            batch_size=batch_size,
        )
    )


@main.command(name="whisperx")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--model", default="small", show_default=True)
@click.option("--language")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--batch_size", "batch_size", type=int)
@click.option("--output_format", "output_format", type=click.Choice(["json", "srt", "vtt", "txt"]), default="json")
@click.option("--output_file", "output_file")
@click.option("--diarize", is_flag=True, default=False)
@click.option("--hf_token")
@click.option("--align_model")
@click.option("--highlight_words", is_flag=True, default=False)
def whisperx(
    input_path: str,
    model: str,
    language: str | None,
    device: str,
    batch_size: int | None,
    output_format: str,
    output_file: str | None,
    diarize: bool,
    hf_token: str | None,
    align_model: str | None,
    highlight_words: bool,
) -> None:
    """WhisperX compatibility frontend."""
    _ = highlight_words
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token
    compat = map_whisperx_options(
        diarize=diarize,
        align_model=align_model,
        output_format=output_format,
        batch_size=batch_size,
    )
    raise SystemExit(
        _run_transcribe(
            input_path=input_path,
            model=model,
            language=language,
            device=device,
            runtime_policy=compat.runtime_policy,
            required_features=(),
            require_alignment=compat.require_alignment,
            require_diarization=compat.require_diarization,
            output_format=compat.output_format,
            output_file=output_file,
            emit_result_json="auto",
            write_failure_json=False,
            diarize_backend=compat.diarize_backend,
            align_backend=compat.align_backend,
            batch_size=compat.batch_size,
        )
    )


@main.command()
@click.option("--json-output", is_flag=True, default=False)
def doctor(json_output: bool) -> None:
    """Inspect the local runtime environment."""
    report = DoctorReport.collect()
    if json_output:
        click.echo(_serialize(report.to_dict()))
        return
    for line in report.to_lines():
        click.echo(line)


@main.group()
def setup() -> None:
    """Setup helpers."""


@setup.command(name="core")
def setup_core() -> None:
    """Show the documented setup path for the core feature set."""
    lines = [
        "Core setup path:",
        "1. python3 -m pip install -e .",
        "2. python3 -m pip install '.[core]'",
        "3. omega doctor",
        "4. omega transcribe sample.wav --output-format json --emit-result-json always",
    ]
    click.echo("\n".join(lines))


@setup.command(name="align")
def setup_align() -> None:
    """Show the documented setup path for forced alignment."""
    lines = [
        "Alignment setup path:",
        "1. python3 -m pip install -e .",
        "2. python3 -m pip install '.[align]'",
        "3. omega doctor",
        "4. omega transcribe sample.wav --require-alignment --align-backend wav2vec2 --emit-result-json always",
        "5. Use a latin-script language or leave --language unset for auto-latin mode",
        "6. For non-latin languages, set OMEGA_ALIGNMENT_ROMANIZER to an external romanizer command",
    ]
    click.echo("\n".join(lines))


@setup.command(name="diarize")
def setup_diarize() -> None:
    """Show the documented setup path for diarization."""
    lines = [
        "Diarization setup path:",
        "1. python3 -m pip install -e .",
        "2. python3 -m pip install '.[diarize]'",
        "3. Ensure ffmpeg and torchcodec are available for audio decode",
        "4. export HF_TOKEN=...",
        "5. omega doctor",
        "6. omega transcribe sample.wav --require-diarization --diarize-backend pyannote --emit-result-json always",
    ]
    click.echo("\n".join(lines))


@setup.command(name="validation")
def setup_validation() -> None:
    """Show the documented setup path for validation assets and reports."""
    lines = [
        "Validation setup path:",
        "1. python3 -m pip install '.[validation]'",
        "2. Prepare dataset folders for D1-D5",
        "3. python3 scripts/export_google_fleurs_fixtures.py D1_SHORT_JA fixtures/d1_short_ja --count 5",
        "4. python3 scripts/export_google_fleurs_fixtures.py D2_SHORT_EN fixtures/d2_short_en --count 5",
        "5. python3 scripts/build_long_fixture.py fixtures/d3_long_mixed/d3_concat_01.wav <wav1> <wav2> ... --gap-ms 500",
        "6. python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/<speaker-a>.wav fixtures/d2_short_en/<speaker-b>.wav fixtures/d4_diarization/d4_mix_01.wav",
        "7. python3 scripts/build_failure_fixtures.py fixtures/d1_short_ja/<source>.wav fixtures/d5_failure_injection",
        "8. python3 scripts/build_dataset_manifest.py <dataset-dir> --dataset-id D1_SHORT_JA",
        "9. python3 scripts/generate_validation_report.py --output validation-report.json",
        "10. Fill docs/VALIDATION_DATASET_MANIFEST.md and docs/BENCHMARK_TEMPLATE.md",
        "11. omega doctor --json-output",
        "12. Run docs/VALIDATION_CHECKLIST.md end-to-end",
    ]
    click.echo("\n".join(lines))
