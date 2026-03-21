from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess

from whisper_omega.runtime.models import BackendError, Segment, Word


_WORD_PATTERN = re.compile(r"[^\s]+")
_LATIN_LANGUAGE_HINTS = {
    "af",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "la",
    "lt",
    "lv",
    "ms",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sq",
    "sv",
    "sw",
    "tl",
    "tr",
    "vi",
}
_HIRAGANA_ROMAJI = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゔ": "vu",
}
_HIRAGANA_DIGRAPHS = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
}


@dataclass(slots=True)
class AlignmentOutcome:
    words: list[Word]
    backend_errors: list[BackendError] = field(default_factory=list)


class AlignmentBackend:
    name = "base"

    def align(
        self,
        audio_path: Path,
        text: str,
        segments: list[Segment],
        words: list[Word],
        language: str | None,
    ) -> AlignmentOutcome:
        raise NotImplementedError

    def capability(self) -> tuple[bool, str | None]:
        return (True, None)


class NoopAlignmentBackend(AlignmentBackend):
    name = "none"

    def align(
        self,
        audio_path: Path,
        text: str,
        segments: list[Segment],
        words: list[Word],
        language: str | None,
    ) -> AlignmentOutcome:
        _ = (audio_path, text, segments, language)
        return AlignmentOutcome(words=words)


class Wav2Vec2AlignmentBackend(AlignmentBackend):
    name = "wav2vec2"

    def capability(self) -> tuple[bool, str | None]:
        try:
            import torchaudio  # noqa: F401
        except Exception:
            return (False, "ALIGNMENT_BACKEND_UNAVAILABLE")
        if os.environ.get("OMEGA_ALIGNMENT_ROMANIZER"):
            return (True, None)
        return (True, None)

    def align(
        self,
        audio_path: Path,
        text: str,
        segments: list[Segment],
        words: list[Word],
        language: str | None,
    ) -> AlignmentOutcome:
        try:
            import torch
            import torchaudio
        except Exception:
            return self._failure("ALIGNMENT_BACKEND_UNAVAILABLE", "dependency", "torchaudio is not installed")

        resolved_language = resolve_alignment_language(language)
        if resolved_language is None:
            return self._failure(
                "ALIGNMENT_LANGUAGE_UNSUPPORTED",
                "validation",
                f"alignment backend does not yet support language={language!r}",
            )

        normalized_words = words or _words_from_segments(segments) or _words_from_text(text)
        if not normalized_words:
            return self._failure("ALIGNMENT_TEXT_UNAVAILABLE", "validation", "alignment requires transcript words")

        try:
            bundle = torchaudio.pipelines.MMS_FA
            tokenizer = bundle.get_tokenizer()
            aligner = bundle.get_aligner()
            model = bundle.get_model()
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:
            return self._failure("ALIGNMENT_MODEL_UNAVAILABLE", "backend", str(exc))

        try:
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
                sample_rate = bundle.sample_rate

            prepared_words = [_prepare_word_for_alignment(word.text, resolved_language) for word in normalized_words]
            if any(not token for token in prepared_words):
                return self._failure(
                    "ALIGNMENT_TEXT_UNSUPPORTED",
                    "validation",
                    "alignment transcript contains unsupported characters",
                )

            token_sequences = tokenizer(prepared_words)
            with torch.inference_mode():
                emission, _ = model(waveform)
            spans = aligner(emission[0], token_sequences)
        except KeyError as exc:
            return self._failure(
                "ALIGNMENT_TEXT_UNSUPPORTED",
                "validation",
                f"alignment tokenizer does not support character: {exc}",
            )
        except Exception as exc:
            return self._failure("ALIGNMENT_RUNTIME_FAILURE", "backend", str(exc))

        try:
            aligned_words = _apply_spans(normalized_words, spans, waveform.shape[-1], sample_rate, emission.shape[1])
        except ValueError as exc:
            return self._failure("ALIGNMENT_RUNTIME_FAILURE", "backend", str(exc))
        return AlignmentOutcome(words=aligned_words)

    def _failure(self, code: str, category: str, message: str) -> AlignmentOutcome:
        return AlignmentOutcome(
            words=[],
            backend_errors=[
                BackendError(
                    backend=self.name,
                    code=code,
                    category=category,
                    message=message,
                    retryable=False,
                )
            ],
        )


class UnavailableWav2Vec2Backend(AlignmentBackend):
    name = "wav2vec2"

    def align(
        self,
        audio_path: Path,
        text: str,
        segments: list[Segment],
        words: list[Word],
        language: str | None,
    ) -> AlignmentOutcome:
        _ = (audio_path, text, segments, language)
        if words:
            return AlignmentOutcome(words=words)
        return AlignmentOutcome(
            words=words,
            backend_errors=[
                BackendError(
                    backend=self.name,
                    code="ALIGNMENT_MODEL_UNAVAILABLE",
                    category="backend",
                    message="alignment backend could not produce word timings",
                    retryable=False,
                )
            ],
        )


def _words_from_segments(segments: list[Segment]) -> list[Word]:
    words: list[Word] = []
    for segment in segments:
        words.extend(
            Word(text=match.group(0), start=segment.start, end=segment.end, speaker=segment.speaker)
            for match in _WORD_PATTERN.finditer(segment.text)
        )
    return words


def _words_from_text(text: str) -> list[Word]:
    return [Word(text=match.group(0), start=0.0, end=0.0, speaker=None) for match in _WORD_PATTERN.finditer(text)]


def _normalize_word(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalpha() or ch in {"'", "*"})


def resolve_alignment_language(language: str | None) -> str | None:
    if language is None or language == "":
        return "auto-latin"
    normalized = language.lower().replace("_", "-")
    base = normalized.split("-", 1)[0]
    if base in _LATIN_LANGUAGE_HINTS:
        return base
    if base == "ja":
        return "ja-kana"
    if os.environ.get("OMEGA_ALIGNMENT_ROMANIZER"):
        return f"romanized:{base}"
    return None


def _prepare_word_for_alignment(text: str, resolved_language: str) -> str:
    if resolved_language == "ja-kana":
        return _romanize_japanese_word(text)
    if resolved_language.startswith("romanized:"):
        return _romanize_word(text)
    return _normalize_word(text)


def _romanize_word(text: str) -> str:
    command = os.environ.get("OMEGA_ALIGNMENT_ROMANIZER")
    if not command:
        return ""
    try:
        completed = subprocess.run(
            command,
            input=text,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
    except Exception:
        return ""
    return _normalize_word(completed.stdout)


def _romanize_kana(text: str) -> str:
    normalized = []
    for char in text.strip():
        if char == "ー":
            normalized.append(char)
            continue
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            normalized.append(chr(codepoint - 0x60))
        else:
            normalized.append(char)
    hira = "".join(normalized)
    if not hira:
        return ""

    result = []
    index = 0
    geminate = False
    while index < len(hira):
        char = hira[index]
        if char in {" ", "\t"}:
            index += 1
            continue
        if char == "っ":
            geminate = True
            index += 1
            continue
        if char == "ー":
            if result:
                last = result[-1]
                vowel = next((letter for letter in reversed(last) if letter in "aeiou"), "")
                if vowel:
                    result.append(vowel)
            index += 1
            continue
        if index + 1 < len(hira):
            digraph = hira[index : index + 2]
            mapped = _HIRAGANA_DIGRAPHS.get(digraph)
            if mapped:
                if geminate and mapped:
                    mapped = mapped[0] + mapped
                    geminate = False
                result.append(mapped)
                index += 2
                continue
        mapped = _HIRAGANA_ROMAJI.get(char)
        if mapped is None:
            return ""
        if geminate and mapped:
            mapped = mapped[0] + mapped
            geminate = False
        result.append(mapped)
        index += 1
    return _normalize_word("".join(result))


def _romanize_japanese_word(text: str) -> str:
    mapped = _japanese_reading_override(text)
    if mapped:
        return _romanize_kana(mapped)
    return _romanize_kana(text)


def _japanese_reading_override(text: str) -> str | None:
    mapping_path = os.environ.get("OMEGA_ALIGNMENT_JA_READING_MAP")
    if not mapping_path:
        return None
    try:
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    reading = payload.get(text)
    if isinstance(reading, str):
        return reading
    return None


def _apply_spans(
    base_words: list[Word],
    spans: list[list],
    num_samples: int,
    sample_rate: int,
    num_frames: int,
) -> list[Word]:
    if len(base_words) != len(spans):
        raise ValueError("alignment span count did not match word count")

    seconds_per_frame = (num_samples / float(sample_rate)) / float(num_frames)
    aligned_words: list[Word] = []
    for word, token_spans in zip(base_words, spans):
        if not token_spans:
            raise ValueError(f"alignment did not return spans for word: {word.text}")
        start = token_spans[0].start * seconds_per_frame
        end = token_spans[-1].end * seconds_per_frame
        confidence = sum(span.score for span in token_spans) / len(token_spans)
        aligned_words.append(
            Word(
                text=word.text,
                start=start,
                end=end,
                speaker=word.speaker,
                confidence=confidence,
            )
        )
    return aligned_words
