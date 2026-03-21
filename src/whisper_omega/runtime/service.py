from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import os
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from whisper_omega.align.base import AlignmentBackend, NoopAlignmentBackend, Wav2Vec2AlignmentBackend
from whisper_omega.asr.base import ASRBackend
from whisper_omega.asr.faster_whisper_backend import FasterWhisperBackend, dependency_error
from whisper_omega.diarize.base import DiarizationBackend, NoopDiarizationBackend, UnavailablePyannoteBackend
from whisper_omega.io.writers import write_json, write_srt, write_txt, write_vtt
from whisper_omega.runtime.models import BackendError, Fallback, Metadata, TranscriptionResult
from whisper_omega.runtime.policy import PolicyConfig, cuda_available, effective_device


@dataclass(slots=True)
class ServiceConfig:
    policy: PolicyConfig
    required_features: list[str] = field(default_factory=list)
    output_format: str = "json"
    emit_result_json: str = "auto"
    write_failure_json: bool = False
    diarize_backend: str = "none"
    align_backend: str = "none"
    model_name: str = "small"
    language: str | None = None
    batch_size: int | None = None


class TranscriptionService:
    def __init__(
        self,
        config: ServiceConfig,
        asr_backend: ASRBackend | None = None,
        alignment_backend: AlignmentBackend | None = None,
        diarization_backend: DiarizationBackend | None = None,
    ) -> None:
        self.config = config
        self.asr_backend = asr_backend or FasterWhisperBackend()
        self.alignment_backend = alignment_backend or self._make_alignment_backend(config.align_backend)
        self.diarization_backend = diarization_backend or self._make_diarization_backend(config.diarize_backend)

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        requested_device = self.config.policy.device
        actual_device = effective_device(requested_device)
        os.environ["OMEGA_AUDIO_PATH"] = str(audio_path)
        os.environ["OMEGA_DEVICE"] = actual_device
        metadata = Metadata(
            asr_backend=self.asr_backend.name,
            align_backend=self.config.align_backend,
            diarization_backend=self.config.diarize_backend,
            device=actual_device,
            requested_device=requested_device,
            actual_device=actual_device,
            alignment_strategy=None,
            alignment_token_source=None,
            requested_features=["asr", *self.config.required_features],
            completed_features=[],
            failed_features=[],
        )

        if requested_device == "cuda" and actual_device != "cuda":
            return self._failure_result(
                metadata=metadata,
                code="GPU_UNAVAILABLE",
                category="runtime",
                message="CUDA device not available",
                backend=self.asr_backend.name,
            )

        if self.config.policy.runtime_policy == "strict-gpu" and actual_device != "cuda":
            return self._failure_result(
                metadata=metadata,
                code="GPU_UNAVAILABLE",
                category="runtime",
                message="strict-gpu policy requires a CUDA device",
                backend=self.asr_backend.name,
            )

        try:
            backend_result = self.asr_backend.transcribe(
                audio_path=audio_path,
                model_name=self.config.model_name,
                language=self.config.language,
                device=actual_device,
                batch_size=self.config.batch_size,
            )
        except RuntimeError as exc:
            marker = str(exc)
            if marker.startswith("DEPENDENCY_MISSING:"):
                message = marker.split(":", 1)[1]
                return self._failure_result(
                    metadata=metadata,
                    code="DEPENDENCY_MISSING",
                    category="dependency",
                    message=f"missing dependency: {message}",
                    backend=self.asr_backend.name,
                )
            return self._failure_result(
                metadata=metadata,
                code="AUDIO_DECODE_FAILURE",
                category="runtime",
                message=str(exc),
                backend=self.asr_backend.name,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return self._failure_result(
                metadata=metadata,
                code="INTERNAL_ERROR",
                category="internal",
                message=str(exc),
                backend=self.asr_backend.name,
            )

        metadata.completed_features.append("asr")
        result = TranscriptionResult(
            schema_version="1.0.0",
            status="success",
            text=backend_result.text,
            language=backend_result.language,
            segments=backend_result.segments,
            words=backend_result.words,
            speakers=[],
            metadata=metadata,
        )
        return self._apply_optional_features(audio_path, result)

    def _apply_optional_features(self, audio_path: Path, result: TranscriptionResult) -> TranscriptionResult:
        failed: list[str] = []
        backend_errors: list[BackendError] = []
        fallbacks: list[Fallback] = []
        words = result.words
        segments = result.segments
        speakers = result.speakers

        if "alignment" in self.config.required_features:
            outcome = self.alignment_backend.align(audio_path, result.text, segments, words, result.language)
            result.metadata.alignment_strategy = outcome.strategy
            result.metadata.alignment_token_source = outcome.token_source
            if outcome.words:
                words = outcome.words
            if outcome.backend_errors:
                failed.append("alignment")
                backend_errors.extend(outcome.backend_errors)
                fallbacks.append(
                    Fallback(
                        type="feature",
                        from_value="alignment",
                        to_value="segment_only",
                        reason=outcome.backend_errors[0].code,
                    )
                )
            else:
                result.metadata.completed_features.append("alignment")

        if "diarization" in self.config.required_features:
            outcome = self.diarization_backend.diarize(segments, words)
            segments = outcome.segments
            words = outcome.words
            speakers = outcome.speakers
            if outcome.backend_errors:
                failed.append("diarization")
                backend_errors.extend(outcome.backend_errors)
                fallbacks.append(
                    Fallback(
                        type="feature",
                        from_value="diarization",
                        to_value="speaker_null",
                        reason=outcome.backend_errors[0].code,
                    )
                )
            else:
                result.metadata.completed_features.append("diarization")

        if not failed:
            return TranscriptionResult(
                schema_version=result.schema_version,
                status=result.status,
                text=result.text,
                language=result.language,
                segments=segments,
                words=words,
                speakers=speakers,
                metadata=result.metadata,
            )

        if self.config.policy.runtime_policy in {"strict", "strict-gpu"}:
            return TranscriptionResult(
                schema_version="1.0.0",
                status="failure",
                text=result.text,
                language=result.language,
                segments=segments,
                words=words,
                speakers=speakers,
                metadata=Metadata(
                    asr_backend=result.metadata.asr_backend,
                    align_backend=result.metadata.align_backend,
                    diarization_backend=result.metadata.diarization_backend,
                    device=result.metadata.device,
                    requested_device=result.metadata.requested_device,
                    actual_device=result.metadata.actual_device,
                    alignment_strategy=result.metadata.alignment_strategy,
                    alignment_token_source=result.metadata.alignment_token_source,
                    fallbacks=[],
                    requested_features=result.metadata.requested_features,
                    completed_features=result.metadata.completed_features,
                    failed_features=failed,
                ),
                error_code=backend_errors[0].code,
                error_category=backend_errors[0].category,
                backend_errors=backend_errors,
            )

        return TranscriptionResult(
            schema_version="1.0.0",
            status="degraded",
            text=result.text,
            language=result.language,
            segments=segments,
            words=words,
            speakers=speakers,
            metadata=Metadata(
                asr_backend=result.metadata.asr_backend,
                align_backend=result.metadata.align_backend,
                diarization_backend=result.metadata.diarization_backend,
                device=result.metadata.device,
                requested_device=result.metadata.requested_device,
                actual_device=result.metadata.actual_device,
                alignment_strategy=result.metadata.alignment_strategy,
                alignment_token_source=result.metadata.alignment_token_source,
                fallbacks=fallbacks,
                requested_features=result.metadata.requested_features,
                completed_features=result.metadata.completed_features,
                failed_features=failed,
            ),
            error_code=backend_errors[0].code,
            error_category=backend_errors[0].category,
            backend_errors=backend_errors,
        )

    @staticmethod
    def _make_alignment_backend(name: str) -> AlignmentBackend:
        if name == "wav2vec2":
            return Wav2Vec2AlignmentBackend()
        return NoopAlignmentBackend()

    @staticmethod
    def _make_diarization_backend(name: str) -> DiarizationBackend:
        if name == "pyannote":
            return UnavailablePyannoteBackend()
        return NoopDiarizationBackend()

    def _failure_result(
        self,
        metadata: Metadata,
        code: str,
        category: str,
        message: str,
        backend: str,
    ) -> TranscriptionResult:
        metadata.failed_features = ["asr"]
        return TranscriptionResult(
            schema_version="1.0.0",
            status="failure",
            text="",
            language=self.config.language or "",
            segments=[],
            words=[],
            speakers=[],
            metadata=metadata,
            error_code=code,
            error_category=category,
            backend_errors=[
                dependency_error(backend, message)
                if category == "dependency"
                else BackendError(
                    backend=backend,
                    code=code,
                    category=category,
                    message=message,
                    retryable=category == "runtime",
                )
            ],
        )

    def write_output(self, result: TranscriptionResult, output_format: str, output_path: Path) -> None:
        writers = {
            "json": write_json,
            "txt": write_txt,
            "srt": write_srt,
            "vtt": write_vtt,
        }
        if result.status == "failure":
            write_json(result, output_path)
            return
        writers[output_format](result, output_path)


@dataclass(slots=True)
class DoctorReport:
    python_version: str
    platform_name: str
    faster_whisper_available: bool
    ctranslate2_available: bool
    torch_available: bool
    torchaudio_available: bool
    torchcodec_available: bool
    torchcodec_importable: bool
    torch_cuda_available: bool
    nvidia_smi_available: bool
    ffmpeg_available: bool
    nvidia_summary: str
    hf_token_configured: bool
    diarization_backend_available: bool
    alignment_backend_available: bool
    diarization_ready: bool
    diarization_issue_code: str | None
    diarization_decode_ready: bool
    diarization_decode_backend: str
    alignment_ready: bool
    alignment_issue_code: str | None
    alignment_romanizer_configured: bool
    alignment_text_map_configured: bool
    alignment_ja_reading_map_configured: bool
    alignment_map_issue_code: str | None
    pyannote_num_speakers: int | None
    pyannote_min_speakers: int | None
    pyannote_max_speakers: int | None
    cache_dir: str
    cache_dir_writable: bool
    detected_device: str
    known_issue_codes: list[str]
    recommended_actions: list[str]

    @classmethod
    def collect(cls) -> "DoctorReport":
        faster_available = _module_available("faster_whisper")
        ctranslate2_available = _module_available("ctranslate2")
        torch_available = _module_available("torch")
        torchaudio_available = _module_available("torchaudio")
        torchcodec_available = _module_available("torchcodec")
        diarization_backend = UnavailablePyannoteBackend()
        alignment_backend = Wav2Vec2AlignmentBackend()
        diarization_backend_available = _module_available("pyannote.audio")
        alignment_backend_available = torchaudio_available
        ffmpeg_available = shutil.which("ffmpeg") is not None
        torchcodec_importable = torchcodec_available and ffmpeg_available and _module_importable("torchcodec")
        diarization_ready, diarization_issue_code = diarization_backend.capability()
        diarization_decode_ready = False
        diarization_decode_backend = "none"
        if torchaudio_available:
            diarization_decode_ready = True
            diarization_decode_backend = "torchaudio"
        elif ffmpeg_available and torchcodec_importable:
            diarization_decode_ready = True
            diarization_decode_backend = "ffmpeg+torchcodec"
        if diarization_ready and not diarization_decode_ready:
            diarization_ready = False
            diarization_issue_code = "DIARIZATION_DECODE_UNAVAILABLE"
        alignment_ready, alignment_issue_code = alignment_backend.capability()
        alignment_romanizer_configured = bool(os.environ.get("OMEGA_ALIGNMENT_ROMANIZER"))
        alignment_text_map_configured = False
        alignment_ja_reading_map_configured = False
        alignment_map_issue_code = None
        text_map_path = os.environ.get("OMEGA_ALIGNMENT_TEXT_MAP")
        if text_map_path:
            text_map = Path(text_map_path)
            if text_map.is_file() and os.access(text_map, os.R_OK):
                alignment_text_map_configured = True
            else:
                alignment_map_issue_code = "ALIGNMENT_TEXT_MAP_INVALID"
        ja_map_path = os.environ.get("OMEGA_ALIGNMENT_JA_READING_MAP")
        if ja_map_path:
            ja_map = Path(ja_map_path)
            if ja_map.is_file() and os.access(ja_map, os.R_OK):
                alignment_ja_reading_map_configured = True
            else:
                alignment_map_issue_code = alignment_map_issue_code or "ALIGNMENT_TEXT_MAP_INVALID"
        pyannote_num_speakers = _int_env("OMEGA_PYANNOTE_NUM_SPEAKERS")
        pyannote_min_speakers = _int_env("OMEGA_PYANNOTE_MIN_SPEAKERS")
        pyannote_max_speakers = _int_env("OMEGA_PYANNOTE_MAX_SPEAKERS")
        torch_cuda = cuda_available() if torch_available else False
        has_nvidia_smi = shutil.which("nvidia-smi") is not None
        nvidia_summary = "not available"
        if has_nvidia_smi:
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                nvidia_summary = completed.stdout.strip() or "available"
            except Exception:
                nvidia_summary = "available but unreadable"
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        cache_dir_writable = os.access(cache_dir, os.W_OK) if cache_dir.exists() else os.access(cache_dir.parent, os.W_OK)
        known_issue_codes = []
        if not faster_available:
            known_issue_codes.append("DEPENDENCY_MISSING")
        if diarization_issue_code:
            known_issue_codes.append(diarization_issue_code)
        if alignment_issue_code:
            known_issue_codes.append(alignment_issue_code)
        if alignment_map_issue_code:
            known_issue_codes.append(alignment_map_issue_code)
        if not cache_dir_writable:
            known_issue_codes.append("OUTPUT_PERMISSION_DENIED")
        recommended_actions = _recommended_actions(known_issue_codes)
        return cls(
            python_version=sys.version.split()[0],
            platform_name=platform.platform(),
            faster_whisper_available=faster_available,
            ctranslate2_available=ctranslate2_available,
            torch_available=torch_available,
            torchaudio_available=torchaudio_available,
            torchcodec_available=torchcodec_available,
            torchcodec_importable=torchcodec_importable,
            torch_cuda_available=torch_cuda,
            nvidia_smi_available=has_nvidia_smi,
            ffmpeg_available=ffmpeg_available,
            nvidia_summary=nvidia_summary,
            hf_token_configured=bool(os.environ.get("HF_TOKEN")),
            diarization_backend_available=diarization_backend_available,
            alignment_backend_available=alignment_backend_available,
            diarization_ready=diarization_ready,
            diarization_issue_code=diarization_issue_code,
            diarization_decode_ready=diarization_decode_ready,
            diarization_decode_backend=diarization_decode_backend,
            alignment_ready=alignment_ready,
            alignment_issue_code=alignment_issue_code,
            alignment_romanizer_configured=alignment_romanizer_configured,
            alignment_text_map_configured=alignment_text_map_configured,
            alignment_ja_reading_map_configured=alignment_ja_reading_map_configured,
            alignment_map_issue_code=alignment_map_issue_code,
            pyannote_num_speakers=pyannote_num_speakers,
            pyannote_min_speakers=pyannote_min_speakers,
            pyannote_max_speakers=pyannote_max_speakers,
            cache_dir=str(cache_dir),
            cache_dir_writable=cache_dir_writable,
            detected_device=effective_device("auto"),
            known_issue_codes=known_issue_codes,
            recommended_actions=recommended_actions,
        )

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "platform": self.platform_name,
            "faster_whisper_available": self.faster_whisper_available,
            "ctranslate2_available": self.ctranslate2_available,
            "torch_available": self.torch_available,
            "torchaudio_available": self.torchaudio_available,
            "torchcodec_available": self.torchcodec_available,
            "torchcodec_importable": self.torchcodec_importable,
            "torch_cuda_available": self.torch_cuda_available,
            "nvidia_smi_available": self.nvidia_smi_available,
            "ffmpeg_available": self.ffmpeg_available,
            "nvidia_summary": self.nvidia_summary,
            "hf_token_configured": self.hf_token_configured,
            "diarization_backend_available": self.diarization_backend_available,
            "alignment_backend_available": self.alignment_backend_available,
            "diarization_ready": self.diarization_ready,
            "diarization_issue_code": self.diarization_issue_code,
            "diarization_decode_ready": self.diarization_decode_ready,
            "diarization_decode_backend": self.diarization_decode_backend,
            "alignment_ready": self.alignment_ready,
            "alignment_issue_code": self.alignment_issue_code,
            "alignment_romanizer_configured": self.alignment_romanizer_configured,
            "alignment_text_map_configured": self.alignment_text_map_configured,
            "alignment_ja_reading_map_configured": self.alignment_ja_reading_map_configured,
            "alignment_map_issue_code": self.alignment_map_issue_code,
            "alignment_language_strategy": "latin-script languages via torchaudio MMS_FA",
            "pyannote_num_speakers": self.pyannote_num_speakers,
            "pyannote_min_speakers": self.pyannote_min_speakers,
            "pyannote_max_speakers": self.pyannote_max_speakers,
            "cache_dir": self.cache_dir,
            "cache_dir_writable": self.cache_dir_writable,
            "detected_device": self.detected_device,
            "known_issue_codes": self.known_issue_codes,
            "recommended_actions": self.recommended_actions,
        }

    def to_lines(self) -> list[str]:
        return [
            f"Python: {self.python_version}",
            f"Platform: {self.platform_name}",
            f"faster-whisper: {'ok' if self.faster_whisper_available else 'missing'}",
            f"ctranslate2: {'ok' if self.ctranslate2_available else 'missing'}",
            f"torch: {'ok' if self.torch_available else 'missing'}",
            f"torchaudio: {'ok' if self.torchaudio_available else 'missing'}",
            f"torchcodec: {'ok' if self.torchcodec_available else 'missing'}",
            f"torchcodec import: {'ok' if self.torchcodec_importable else 'unavailable'}",
            f"torch CUDA: {'ok' if self.torch_cuda_available else 'unavailable'}",
            f"nvidia-smi: {'ok' if self.nvidia_smi_available else 'missing'}",
            f"ffmpeg: {'ok' if self.ffmpeg_available else 'missing'}",
            f"NVIDIA: {self.nvidia_summary}",
            f"HF_TOKEN: {'configured' if self.hf_token_configured else 'missing'}",
            f"diarization backend: {'ready' if self.diarization_ready else self.diarization_issue_code or 'missing'}",
            f"diarization decode stack: {self.diarization_decode_backend if self.diarization_decode_ready else 'incomplete'}",
            f"alignment backend: {'ready' if self.alignment_ready else self.alignment_issue_code or 'missing'}",
            f"alignment romanizer: {'configured' if self.alignment_romanizer_configured else 'not configured'}",
            f"alignment text map: {'configured' if self.alignment_text_map_configured else self.alignment_map_issue_code or 'not configured'}",
            f"alignment ja map: {'configured' if self.alignment_ja_reading_map_configured else self.alignment_map_issue_code or 'not configured'}",
            "alignment strategy: latin-script languages via torchaudio MMS_FA",
            f"pyannote speaker hints: num={self.pyannote_num_speakers} min={self.pyannote_min_speakers} max={self.pyannote_max_speakers}",
            f"cache dir: {self.cache_dir} ({'writable' if self.cache_dir_writable else 'not writable'})",
            f"auto device: {self.detected_device}",
            f"known issues: {', '.join(self.known_issue_codes) if self.known_issue_codes else 'none'}",
            f"recommended actions: {' | '.join(self.recommended_actions) if self.recommended_actions else 'none'}",
        ]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _module_importable(module_name: str, timeout_seconds: float = 2.0) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
        )
        return True
    except Exception:
        return False


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _recommended_actions(issue_codes: list[str]) -> list[str]:
    actions: list[str] = []
    if "DEPENDENCY_MISSING" in issue_codes:
        actions.append("Install the core extra with `python3 -m pip install '.[core]'`.")
    if "HF_TOKEN_MISSING" in issue_codes:
        actions.append("Set `HF_TOKEN` before running pyannote diarization.")
    if "AUDIO_DECODE_FAILURE" in issue_codes or "DIARIZATION_DECODE_UNAVAILABLE" in issue_codes:
        actions.append("Install system FFmpeg and verify `torchcodec` can import successfully.")
    if "ALIGNMENT_BACKEND_UNAVAILABLE" in issue_codes:
        actions.append("Install the align extra with `python3 -m pip install '.[align]'`.")
    if "OUTPUT_PERMISSION_DENIED" in issue_codes:
        actions.append("Choose a writable cache/output directory.")
    return actions
