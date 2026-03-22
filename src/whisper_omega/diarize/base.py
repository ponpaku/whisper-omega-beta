from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
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

        speaker_turns = list(_iter_speaker_turns(diarization))
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
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        yield (float(turn.start), float(turn.end), str(speaker))


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
