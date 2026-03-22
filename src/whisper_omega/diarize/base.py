from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import os
import tempfile
from typing import Iterable
import wave

from whisper_omega.runtime.models import BackendError, Segment, Speaker, Word


@dataclass(slots=True)
class DiarizationOutcome:
    segments: list[Segment]
    words: list[Word]
    speakers: list[Speaker]
    backend_errors: list[BackendError] = field(default_factory=list)


class DiarizationBackend:
    name = "base"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        raise NotImplementedError

    def capability(self) -> tuple[bool, str | None]:
        return (True, None)


class NoopDiarizationBackend(DiarizationBackend):
    name = "none"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        return DiarizationOutcome(segments=segments, words=words, speakers=[])


@dataclass(slots=True)
class _StereoChannelAnalysis:
    sample_rate: int
    duration: float
    left_prefix: list[float]
    right_prefix: list[float]


class _ChannelValidationError(ValueError):
    pass


class _ChannelRuntimeError(RuntimeError):
    pass


class ChannelDiarizationBackend(DiarizationBackend):
    name = "channel"

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        audio_path = os.environ.get("OMEGA_AUDIO_PATH")
        if not audio_path:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="CONFIG_INVALID",
                category="configuration",
                message="OMEGA_AUDIO_PATH is required for channel diarization",
                retryable=False,
            )

        try:
            analysis = _load_stereo_channel_analysis(audio_path)
        except _ChannelValidationError as exc:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="DIARIZATION_CHANNELS_UNAVAILABLE",
                category="validation",
                message=str(exc),
                retryable=False,
            )
        except _ChannelRuntimeError as exc:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="DIARIZATION_AUDIO_UNSUPPORTED",
                category="runtime",
                message=str(exc),
                retryable=False,
            )

        assigned_segments = [
            _with_speaker(segment, _speaker_for_stereo_interval(segment.start, segment.end, analysis))
            for segment in segments
        ]
        assigned_words = [
            _with_word_speaker(word, _speaker_for_stereo_interval(word.start, word.end, analysis))
            for word in words
        ]
        used_speakers = {
            speaker
            for speaker in [*(segment.speaker for segment in assigned_segments), *(word.speaker for word in assigned_words)]
            if speaker is not None
        }
        if not used_speakers:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="DIARIZATION_CHANNEL_AMBIGUOUS",
                category="validation",
                message="channel diarization could not confidently assign a speaker from stereo energy",
                retryable=False,
            )

        speakers = [
            Speaker(id=speaker_id, start=0.0, end=analysis.duration, label=_channel_label(speaker_id))
            for speaker_id in ("CHANNEL_LEFT", "CHANNEL_RIGHT")
            if speaker_id in used_speakers
        ]
        return DiarizationOutcome(
            segments=assigned_segments,
            words=assigned_words,
            speakers=speakers,
        )


class NemoDiarizationBackend(DiarizationBackend):
    name = "nemo"

    def capability(self) -> tuple[bool, str | None]:
        if importlib.util.find_spec("nemo") is None:
            return (False, "DIARIZATION_BACKEND_UNAVAILABLE")
        if importlib.util.find_spec("omegaconf") is None:
            return (False, "DIARIZATION_BACKEND_UNAVAILABLE")
        config_path = os.environ.get("OMEGA_NEMO_CONFIG")
        if config_path and not os.access(config_path, os.R_OK):
            return (False, "CONFIG_INVALID")
        return (True, None)

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        audio_path = os.environ.get("OMEGA_AUDIO_PATH")
        if not audio_path:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="CONFIG_INVALID",
                category="configuration",
                message="OMEGA_AUDIO_PATH is required for nemo diarization",
                retryable=False,
            )

        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except Exception:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="DIARIZATION_BACKEND_UNAVAILABLE",
                category="dependency",
                message="nemo diarization dependencies are not installed",
                retryable=False,
            )

        try:
            with tempfile.TemporaryDirectory(prefix="omega-nemo-") as tmpdir:
                tmp_root = os.path.abspath(tmpdir)
                manifest_path = os.path.join(tmp_root, "manifest.json")
                out_dir = os.path.join(tmp_root, "out")
                os.makedirs(out_dir, exist_ok=True)
                _write_nemo_manifest(audio_path, manifest_path)
                config = _build_nemo_config(OmegaConf, manifest_path, out_dir, device=os.environ.get("OMEGA_DEVICE", "cpu"))
                diarizer = ClusteringDiarizer(cfg=config)
                diarizer.diarize(paths2audio_files=[audio_path], batch_size=1)
                speaker_turns = _parse_nemo_rttm(out_dir, audio_path)
        except ValueError as exc:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="CONFIG_INVALID",
                category="configuration",
                message=str(exc),
                retryable=False,
            )
        except FileNotFoundError as exc:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="NEMO_OUTPUT_MISSING",
                category="runtime",
                message=str(exc),
                retryable=False,
            )
        except Exception as exc:
            code, category = _classify_nemo_exception(exc)
            return _failure_outcome(
                self.name,
                segments,
                words,
                code=code,
                category=category,
                message=str(exc),
                retryable=category == "runtime",
            )

        if not speaker_turns:
            return _failure_outcome(
                self.name,
                segments,
                words,
                code="NEMO_OUTPUT_MISSING",
                category="runtime",
                message="nemo diarization did not produce RTTM speaker turns",
                retryable=False,
            )

        assigned_segments = [_with_speaker(segment, _speaker_for_interval(segment.start, segment.end, speaker_turns)) for segment in segments]
        assigned_words = [_with_word_speaker(word, _speaker_for_interval(word.start, word.end, speaker_turns)) for word in words]
        speakers = _speakers_from_turns(speaker_turns)
        return DiarizationOutcome(
            segments=assigned_segments,
            words=assigned_words,
            speakers=speakers,
        )


class UnavailablePyannoteBackend(DiarizationBackend):
    name = "pyannote"

    def capability(self) -> tuple[bool, str | None]:
        if importlib.util.find_spec("pyannote.audio") is None:
            return (False, "DIARIZATION_BACKEND_UNAVAILABLE")
        if not os.environ.get("HF_TOKEN"):
            return (False, "HF_TOKEN_MISSING")
        return (True, None)

    def diarize(self, segments: list[Segment], words: list[Word]) -> DiarizationOutcome:
        try:
            from pyannote.audio import Pipeline
        except Exception:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="DIARIZATION_BACKEND_UNAVAILABLE",
                        category="dependency",
                        message="pyannote.audio is not installed",
                        retryable=False,
                    )
                ],
            )
        if not os.environ.get("HF_TOKEN"):
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="HF_TOKEN_MISSING",
                        category="configuration",
                        message="HF_TOKEN is required for pyannote diarization",
                        retryable=False,
                    )
                ],
            )
        audio_path = os.environ.get("OMEGA_AUDIO_PATH")
        if not audio_path:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="CONFIG_INVALID",
                        category="configuration",
                        message="OMEGA_AUDIO_PATH is required for pyannote diarization",
                        retryable=False,
                    )
                ],
            )

        model_id = os.environ.get("OMEGA_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        device = os.environ.get("OMEGA_DEVICE", "cpu")
        try:
            pipeline_kwargs = _pyannote_runtime_kwargs()
        except ValueError as exc:
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code="CONFIG_INVALID",
                        category="configuration",
                        message=str(exc),
                        retryable=False,
                    )
                ],
            )
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
            if hasattr(pipeline, "to"):
                import torch

                torch_device = torch.device("cuda" if device == "cuda" else "cpu")
                pipeline.to(torch_device)
            diarization_input = _load_audio_for_pyannote(audio_path)
            diarization = pipeline(diarization_input, **pipeline_kwargs)
            speaker_turns = list(_iter_speaker_turns(diarization))
        except Exception as exc:
            code, category = _classify_pyannote_exception(exc)
            return DiarizationOutcome(
                segments=segments,
                words=words,
                speakers=[],
                backend_errors=[
                    BackendError(
                        backend=self.name,
                        code=code,
                        category=category,
                        message=str(exc),
                        retryable=category == "runtime",
                    )
                ],
            )

        if not speaker_turns:
            return DiarizationOutcome(segments=segments, words=words, speakers=[])

        assigned_segments = [_with_speaker(segment, _speaker_for_interval(segment.start, segment.end, speaker_turns)) for segment in segments]
        assigned_words = [_with_word_speaker(word, _speaker_for_interval(word.start, word.end, speaker_turns)) for word in words]
        speakers = [
            Speaker(id=speaker, start=start, end=end, label=speaker)
            for start, end, speaker in speaker_turns
        ]
        return DiarizationOutcome(
            segments=assigned_segments,
            words=assigned_words,
            speakers=speakers,
        )


def _iter_speaker_turns(diarization) -> Iterable[tuple[float, float, str]]:
    annotation = _resolve_pyannote_annotation(diarization)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        yield (float(turn.start), float(turn.end), str(speaker))


def _resolve_pyannote_annotation(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization

    annotation = getattr(diarization, "speaker_diarization", None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation

    raise TypeError("pyannote diarization output does not expose speaker turns")


def _load_audio_for_pyannote(audio_path: str):
    try:
        import torchaudio
    except Exception:
        return audio_path

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception:
        return audio_path
    return {"waveform": waveform, "sample_rate": sample_rate}


def _pyannote_runtime_kwargs() -> dict[str, int]:
    mapping = {
        "OMEGA_PYANNOTE_NUM_SPEAKERS": "num_speakers",
        "OMEGA_PYANNOTE_MIN_SPEAKERS": "min_speakers",
        "OMEGA_PYANNOTE_MAX_SPEAKERS": "max_speakers",
    }
    kwargs: dict[str, int] = {}
    for env_name, key in mapping.items():
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer") from exc
        if parsed <= 0:
            raise ValueError(f"{env_name} must be greater than 0")
        kwargs[key] = parsed
    min_speakers = kwargs.get("min_speakers")
    max_speakers = kwargs.get("max_speakers")
    num_speakers = kwargs.get("num_speakers")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise ValueError("OMEGA_PYANNOTE_MIN_SPEAKERS cannot be greater than OMEGA_PYANNOTE_MAX_SPEAKERS")
    if num_speakers and min_speakers and num_speakers < min_speakers:
        raise ValueError("OMEGA_PYANNOTE_NUM_SPEAKERS cannot be smaller than OMEGA_PYANNOTE_MIN_SPEAKERS")
    if num_speakers and max_speakers and num_speakers > max_speakers:
        raise ValueError("OMEGA_PYANNOTE_NUM_SPEAKERS cannot be greater than OMEGA_PYANNOTE_MAX_SPEAKERS")
    return kwargs


def _classify_pyannote_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    if any(token in message for token in ("401", "403", "unauthorized", "forbidden", "invalid token", "authentication", "gated")):
        return ("DIARIZATION_AUTH_FAILURE", "configuration")
    if any(token in message for token in ("ffmpeg", "decode", "codec", "unsupported audio", "could not open audio")):
        return ("DIARIZATION_DECODE_FAILURE", "runtime")
    if any(token in message for token in ("not found", "repository", "revision", "model", "does not exist")):
        return ("DIARIZATION_MODEL_UNAVAILABLE", "backend")
    return ("DIARIZATION_BACKEND_UNAVAILABLE", "backend")


def _classify_nemo_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    if any(token in message for token in ("config", "omegaconf", "yaml")):
        return ("CONFIG_INVALID", "configuration")
    if any(token in message for token in ("not found", "model", "checkpoint", "pretrained")):
        return ("NEMO_MODEL_UNAVAILABLE", "backend")
    return ("NEMO_RUNTIME_FAILURE", "runtime")


def _write_nemo_manifest(audio_path: str, manifest_path: str) -> None:
    payload = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": _nemo_num_speakers(),
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_nemo_config(omegaconf, manifest_path: str, out_dir: str, device: str = "cpu"):
    num_speakers = _nemo_num_speakers()
    max_speakers = _nemo_max_speakers()
    if num_speakers is not None:
        max_speakers = min(max_speakers, num_speakers)

    config_data = {
        "device": device,
        "sample_rate": 16000,
        "batch_size": 1,
        "num_workers": 0,
        "verbose": False,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": out_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": os.environ.get("OMEGA_NEMO_VAD_MODEL", "vad_multilingual_marblenet"),
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.5,
                    "offset": 0.5,
                    "pad_onset": 0.0,
                    "pad_offset": 0.0,
                    "min_duration_on": 0.0,
                    "min_duration_off": 0.0,
                    "filter_speech_first": True,
                    "scale": "absolute",
                },
            },
            "speaker_embeddings": {
                "model_path": os.environ.get("OMEGA_NEMO_SPEAKER_MODEL", "titanet_large"),
                "parameters": {
                    "window_length_in_sec": 1.5,
                    "shift_length_in_sec": 0.75,
                    "multiscale_weights": None,
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": max_speakers,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                }
            },
        }
    }
    config = omegaconf.create(config_data)
    user_config_path = os.environ.get("OMEGA_NEMO_CONFIG")
    if user_config_path:
        if not os.access(user_config_path, os.R_OK):
            raise ValueError("OMEGA_NEMO_CONFIG must point to a readable yaml file")
        config = omegaconf.merge(config, omegaconf.load(user_config_path))
    return config


def _nemo_num_speakers() -> int | None:
    value = os.environ.get("OMEGA_NEMO_NUM_SPEAKERS") or os.environ.get("OMEGA_PYANNOTE_NUM_SPEAKERS")
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("OMEGA_NEMO_NUM_SPEAKERS must be an integer") from exc
    if parsed <= 0:
        raise ValueError("OMEGA_NEMO_NUM_SPEAKERS must be greater than 0")
    return parsed


def _nemo_max_speakers() -> int:
    value = os.environ.get("OMEGA_NEMO_MAX_SPEAKERS") or os.environ.get("OMEGA_PYANNOTE_MAX_SPEAKERS") or "8"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("OMEGA_NEMO_MAX_SPEAKERS must be an integer") from exc
    if parsed <= 0:
        raise ValueError("OMEGA_NEMO_MAX_SPEAKERS must be greater than 0")
    return parsed


def _parse_nemo_rttm(out_dir: str, audio_path: str) -> list[tuple[float, float, str]]:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    candidates: list[str] = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            if not name.endswith(".rttm"):
                continue
            if os.path.splitext(name)[0] == stem:
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"nemo diarization did not produce an RTTM for {stem}")
    return _read_rttm_turns(candidates[0])


def _read_rttm_turns(path: str) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            turns.append((start, start + duration, speaker))
    return turns


def _speakers_from_turns(speaker_turns: list[tuple[float, float, str]]) -> list[Speaker]:
    bounds: dict[str, tuple[float, float]] = {}
    for start, end, speaker in speaker_turns:
        if speaker not in bounds:
            bounds[speaker] = (start, end)
            continue
        current_start, current_end = bounds[speaker]
        bounds[speaker] = (min(current_start, start), max(current_end, end))
    return [
        Speaker(id=speaker, start=start, end=end, label=speaker)
        for speaker, (start, end) in sorted(bounds.items())
    ]


def _load_stereo_channel_analysis(audio_path: str) -> _StereoChannelAnalysis:
    try:
        with wave.open(audio_path, "rb") as handle:
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            frame_bytes = handle.readframes(frame_count)
    except (FileNotFoundError, wave.Error, OSError) as exc:
        raise _ChannelRuntimeError(f"could not read stereo audio for channel diarization: {exc}") from exc

    if channel_count < 2:
        raise _ChannelValidationError("channel diarization requires a stereo wav input")
    if sample_width not in (1, 2, 3, 4):
        raise _ChannelRuntimeError("channel diarization only supports PCM wav sample widths up to 32-bit")
    if sample_rate <= 0:
        raise _ChannelRuntimeError("channel diarization requires a positive sample rate")

    frame_width = channel_count * sample_width
    left_prefix = [0.0]
    right_prefix = [0.0]
    for offset in range(0, len(frame_bytes), frame_width):
        frame = frame_bytes[offset : offset + frame_width]
        if len(frame) < frame_width:
            break
        left_sample = _pcm_sample_to_float(frame[0:sample_width], sample_width)
        right_sample = _pcm_sample_to_float(frame[sample_width : sample_width * 2], sample_width)
        left_prefix.append(left_prefix[-1] + abs(left_sample))
        right_prefix.append(right_prefix[-1] + abs(right_sample))

    duration = frame_count / sample_rate if frame_count else 0.0
    return _StereoChannelAnalysis(
        sample_rate=sample_rate,
        duration=duration,
        left_prefix=left_prefix,
        right_prefix=right_prefix,
    )


def _pcm_sample_to_float(data: bytes, sample_width: int) -> float:
    if sample_width == 1:
        return float(data[0] - 128)
    return float(int.from_bytes(data, byteorder="little", signed=True))


def _speaker_for_stereo_interval(start: float, end: float, analysis: _StereoChannelAnalysis) -> str | None:
    left_energy = _interval_energy(analysis.left_prefix, analysis.sample_rate, start, end)
    right_energy = _interval_energy(analysis.right_prefix, analysis.sample_rate, start, end)
    strongest = max(left_energy, right_energy)
    if strongest <= 0:
        return None
    if abs(left_energy - right_energy) / strongest < 0.1:
        return None
    if left_energy > right_energy:
        return "CHANNEL_LEFT"
    return "CHANNEL_RIGHT"


def _interval_energy(prefix: list[float], sample_rate: int, start: float, end: float) -> float:
    if len(prefix) <= 1:
        return 0.0
    frame_count = len(prefix) - 1
    start_index = max(0, min(frame_count - 1, int(start * sample_rate)))
    end_index = max(start_index + 1, min(frame_count, int(end * sample_rate)))
    return prefix[end_index] - prefix[start_index]


def _channel_label(speaker_id: str) -> str:
    if speaker_id == "CHANNEL_LEFT":
        return "Left channel"
    return "Right channel"


def _speaker_for_interval(start: float, end: float, speaker_turns: list[tuple[float, float, str]]) -> str | None:
    midpoint = (start + end) / 2
    best_overlap = 0.0
    best_speaker: str | None = None
    for turn_start, turn_end, speaker in speaker_turns:
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
        elif best_speaker is None and turn_start <= midpoint <= turn_end:
            best_speaker = speaker
    return best_speaker


def _with_speaker(segment: Segment, speaker: str | None) -> Segment:
    return Segment(
        id=segment.id,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        speaker=speaker,
    )


def _with_word_speaker(word: Word, speaker: str | None) -> Word:
    return Word(
        text=word.text,
        start=word.start,
        end=word.end,
        speaker=speaker,
        confidence=word.confidence,
    )


def _failure_outcome(
    backend: str,
    segments: list[Segment],
    words: list[Word],
    *,
    code: str,
    category: str,
    message: str,
    retryable: bool,
) -> DiarizationOutcome:
    return DiarizationOutcome(
        segments=segments,
        words=words,
        speakers=[],
        backend_errors=[
            BackendError(
                backend=backend,
                code=code,
                category=category,
                message=message,
                retryable=retryable,
            )
        ],
    )
