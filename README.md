# whisper-omega

`whisper-omega` is a Whisper transcription CLI focused on predictable runtime behavior, machine-readable failures, and a WhisperX-compatible user experience where it makes sense.

It is designed for people who want:

- a single `transcribe` entrypoint with stable JSON output
- explicit `success` / `degraded` / `failure` outcomes
- clear runtime policy for CPU / GPU behavior
- optional alignment and diarization without silent fallbacks
- a compatibility frontend for common WhisperX-style flags

## What It Currently Provides

- `omega transcribe`
- `omega whisperx`
- `omega doctor`
- `omega setup core`
- JSON / SRT / VTT / TXT writers
- runtime policy: `permissive`, `strict`, `strict-gpu`
- optional `faster-whisper` integration
- optional `torchaudio` forced alignment integration
- built-in stereo `channel` diarization backend for wav inputs
- optional `nemo` diarization integration
- optional `pyannote.audio` diarization integration
- optional `custom` diarization backend via an external command
- WhisperX compatibility mapping for `--model`, `--device`, `--language`, `--batch_size`, `--output_format`, `--diarize`, `--align_model`, `--hf_token`, and `--highlight_words`

## Status

The repository is in an MVP-but-usable state.

- core local acceptance is green
- `pyannote` acceptance is green when `HF_TOKEN` is configured and the required gated models have been accepted
- `nemo` acceptance is green on the reproducible CPU-fixed acceptance path
- GPU acceptance is green on the current validation setup

For the latest reproducible validation commands and expectations, see `docs/VALIDATION_CHECKLIST.md`.

## Quick Start

Create a local environment and install the package:

```bash
python3 -m venv --system-site-packages .venv-system
./.venv-system/bin/python -m pip install -e . --no-build-isolation
./.venv-system/bin/python -m pip install '.[core]' --no-build-isolation
```

Install optional extras as needed:

```bash
./.venv-system/bin/python -m pip install '.[align]' --no-build-isolation
./.venv-system/bin/python -m pip install '.[diarize]' --no-build-isolation
./.venv-system/bin/python -m pip install '.[diarize-nemo]' --no-build-isolation
./.venv-system/bin/python -m pip install '.[validation]' --no-build-isolation
```

Run a quick health check:

```bash
PYTHONPATH=src ./.venv-system/bin/python -m whisper_omega doctor --json-output
```

Try a basic transcription:

```bash
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --output-format json \
  --emit-result-json always
```

For lower-latency or lighter JSON output, you can disable word timestamps and suppress detailed arrays:

```bash
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --no-word-timestamps \
  --no-include-words \
  --no-include-segments \
  --output-format json \
  --emit-result-json always
```

Result JSON always includes `metadata.timings` with `total_ms`, `asr_ms`, `alignment_ms`, `diarization_ms`, `audio_duration_ms`, and `real_time_factor`.

Run the canonical local acceptance flow:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src ./.venv-system/bin/python scripts/run_acceptance.py
```

If you are in an offline or sandboxed environment, `--no-build-isolation` is often the safest install path.

## Common Usage

Forced alignment:

```bash
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device cpu \
  --model small \
  --require-alignment \
  --align-backend wav2vec2 \
  --output-format json \
  --emit-result-json always
```

Stereo channel diarization:

```bash
PYTHONPATH=src ./.venv-system/bin/omega transcribe stereo_sample.wav \
  --device cpu \
  --model small \
  --require-diarization \
  --diarize-backend channel \
  --output-format json \
  --emit-result-json always
```

Pyannote diarization:

```bash
export HF_TOKEN=...
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --language ja \
  --require-diarization \
  --diarize-backend pyannote \
  --output-format json \
  --emit-result-json always
```

NeMo diarization:

```bash
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --require-diarization \
  --diarize-backend nemo \
  --output-format json \
  --emit-result-json always
```

Custom diarization backend:

```bash
export OMEGA_CUSTOM_DIARIZATION_COMMAND="python /path/to/custom_diarizer.py"
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --require-diarization \
  --diarize-backend custom \
  --output-format json \
  --emit-result-json always
```

## Runtime Notes

`omega doctor` is the best first stop when something is missing. It reports:

- detected device and CUDA visibility
- available diarization backends
- alignment readiness
- decode backend readiness
- canonical issue codes and recommended actions

The result JSON contract is designed so automation can distinguish:

- hard failures
- expected degraded paths
- missing optional capabilities

## Diarization Notes

For zero-extra-dependency diarization on stereo wav files, run `omega transcribe --require-diarization --diarize-backend channel ...`. This backend assigns `CHANNEL_LEFT` / `CHANNEL_RIGHT` speakers from per-channel energy and is useful when each speaker is isolated to one stereo side.

For NeMo diarization, install `.[diarize-nemo]` and run `omega transcribe --require-diarization --diarize-backend nemo ...`. You can optionally point `OMEGA_NEMO_CONFIG` at a NeMo diarizer YAML, or set `OMEGA_NEMO_NUM_SPEAKERS` / `OMEGA_NEMO_MAX_SPEAKERS` to steer clustering without a custom config. The backend now normalizes stereo / non-16 kHz inputs to a temporary mono 16 kHz wav before invoking NeMo, which avoids a common VAD collation failure on long-form meeting audio.

For pyannote diarization, set `HF_TOKEN` before running `omega transcribe --require-diarization --diarize-backend pyannote ...`. The backend prefers in-memory waveform loading via `torchaudio` when available, which helps avoid some decode-stack issues. You can also provide `OMEGA_PYANNOTE_NUM_SPEAKERS`, `OMEGA_PYANNOTE_MIN_SPEAKERS`, or `OMEGA_PYANNOTE_MAX_SPEAKERS` to constrain the diarization search space.

For both `pyannote` and `nemo`, the CLI also accepts `--num-speakers`, `--min-speakers`, and `--max-speakers` as backend hints.

Recommended diarization stack is `pyannote.audio` + `HF_TOKEN` + `torchaudio`; if `torchaudio` is unavailable, fall back to `ffmpeg` + `torchcodec`.

Current pyannote gated-model chain observed in validation is `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1`; the Hugging Face account behind `HF_TOKEN` must have accepted each model's user conditions.

If you want a step-by-step setup and troubleshooting guide for `pyannote`, see `docs/PYANNOTE_SETUP.md`.

Known diarization failure families are now split into `HF_TOKEN_MISSING` / `DIARIZATION_AUTH_FAILURE` / `DIARIZATION_MODEL_UNAVAILABLE` / `DIARIZATION_DECODE_FAILURE` / `CONFIG_INVALID` for pyannote, `NEMO_MODEL_UNAVAILABLE` / `NEMO_RUNTIME_FAILURE` / `NEMO_OUTPUT_MISSING` for NeMo, plus `DIARIZATION_CHANNELS_UNAVAILABLE` / `DIARIZATION_CHANNEL_AMBIGUOUS` / `DIARIZATION_AUDIO_UNSUPPORTED` for the built-in channel backend.

The `custom` backend is enabled by `OMEGA_CUSTOM_DIARIZATION_COMMAND`. The command receives a JSON request on stdin with `audio_path`, `segments`, `words`, and `requested_device`, and should return either `speaker_turns` or full `segments` / `words` / `speakers` JSON.

## Alignment Notes

For forced alignment, install the `align` extra and run `omega transcribe --require-alignment --align-backend wav2vec2 ...`.

Current alignment coverage includes latin-script languages plus kana-only Japanese; other unsupported transcripts return a machine-readable alignment validation failure instead of silently degrading.

For Japanese words that include kanji, you can set `OMEGA_ALIGNMENT_JA_READING_MAP` to a JSON file that maps transcript words to kana readings before alignment.

For any language, `OMEGA_ALIGNMENT_TEXT_MAP` can provide a generic JSON word->normalized-token override before the backend tokenizer runs.

When alignment runs, result JSON metadata now records both `alignment_strategy` and `alignment_token_source` so success, fallback, and degraded paths remain inspectable after the fact.

`scripts/build_ja_reading_map.py` can generate a starter JSON map from a Japanese fixture `manifest.json` so you only need to fill in the readings.

`scripts/build_alignment_text_map.py` can generate a generic starter JSON map for any manifest that includes non-latin tokens.

If you have an external romanizer, set `OMEGA_ALIGNMENT_ROMANIZER` and other non-latin transcripts can be pre-romanized before alignment.

Alignment language/token resolution currently follows this order:

1. `OMEGA_ALIGNMENT_TEXT_MAP` if a per-word override exists
2. native latin-script tokenization through `torchaudio` `MMS_FA`
3. built-in kana romanization for Japanese kana-only text
4. `OMEGA_ALIGNMENT_JA_READING_MAP` when Japanese words need reading overrides
5. `OMEGA_ALIGNMENT_ROMANIZER` for other non-latin languages
6. otherwise `ALIGNMENT_LANGUAGE_UNSUPPORTED`

Unsupported alignment inputs now fail machine-readably with `ALIGNMENT_LANGUAGE_UNSUPPORTED`, `ALIGNMENT_TEXT_UNSUPPORTED`, or map validation codes instead of silently falling back.

## Validation

- `scripts/run_acceptance.py` is the canonical local acceptance entrypoint
- `scripts/run_pyannote_acceptance.py` covers environment-bound pyannote validation
- `scripts/run_nemo_acceptance.py` covers environment-bound NeMo validation and forces a CPU-only path for reproducible acceptance
- `scripts/run_gpu_acceptance.py` records `device=auto`, `device=cuda`, and `strict-gpu` behavior
- `scripts/generate_validation_report.py` can aggregate validation output into `validation-report.json`

Current local validation status is: canonical acceptance green, `pyannote_acceptance` green with `HF_TOKEN`, `nemo_acceptance` green on the CPU-fixed acceptance path, and GPU acceptance green on the current validation setup.

## Source Documents

Use these documents as the current source of truth:

- Product and acceptance intent: `whisper-omega-plan/requirements/whisper_omega_v05.md`
- Formal specs: `whisper-omega-plan/specs/appendix_*.md`
- Canonical local acceptance flow: `docs/VALIDATION_CHECKLIST.md`
- Current fixed fixture roster and hashes: `docs/VALIDATION_DATASET_MANIFEST.md`
- Current benchmark and GPU residual-risk snapshot: `docs/BENCHMARK_TEMPLATE.md`
- Current implementation/release status: `IMPLEMENTATION_TASKS.md`
- Open and temporary decisions: `DECISIONS.md`

`whisper-omega-plan/タスクリスト_20260322.md` is best treated as a working snapshot, not the long-term source of truth.

## WhisperX Compatibility

The `omega whisperx` compatibility frontend also accepts `--hf_token` and maps `--align_model` to alignment-required execution.
To keep WhisperX compatibility stable, `omega whisperx --diarize` continues to target the `pyannote` backend by default. Use `omega transcribe --diarize-backend channel|nemo|pyannote` when you want to choose among the expanded backend set explicitly.

| WhisperX flag | whisper-omega status | Notes |
|---|---|---|
| `--model` | supported | Same meaning |
| `--device` | partial | `auto` prefers CUDA when available; `cuda` means GPU required |
| `--language` | supported | Same meaning |
| `--batch_size` | partial | Accepted on `omega whisperx`, mapped to `--batch-size` internally |
| `--output_format` | partial | Accepted on `omega whisperx`, mapped to `--output-format` internally |
| `--diarize` | partial | Requests diarization and selects `pyannote` backend |
| `--align_model` | partial | Requests alignment and selects `wav2vec2` backend |
| `--hf_token` | partial | Sets `HF_TOKEN` when not already configured |
| `--highlight_words` | partial | Accepted for compatibility, prints a warning, and currently has no effect |

Full details live in `whisper-omega-plan/specs/appendix_b_compat_v01.md`.

Compatibility handling rules are:

1. Same semantics: supported silently
2. Mapped but behaviorally different: partial compatibility, optionally with a warning
3. Accepted but currently no-op: partial compatibility with a warning
4. Unsatisfied hard requirements: usage error or machine-readable failure, never silent ignore

## Notes

- Real transcription requires the optional `faster-whisper` dependency.
- Without it, `omega transcribe` returns a machine-readable dependency failure instead of crashing.
- `omega doctor` now reports known issue codes and recommended actions for missing runtime pieces.
- `omega doctor` also reports available diarization backends, canonical issue codes, and structured `backend_statuses` for diarization / decode / alignment readiness.
- `omega doctor` reports whether alignment maps and pyannote speaker-hint env vars are configured, so map-assisted non-latin routing can be checked before runtime.
- `scripts/run_alignment_smoke.py` runs fixture-backed alignment routing checks for `D1_SHORT_JA` and `D2_SHORT_EN`.
- `scripts/run_diarization_smoke.py` runs fixture-backed diarization assignment checks for the current D4 mixes across both `pyannote` and `nemo` smoke backends.
- `docs/VALIDATION_DATASET_CANDIDATES.md` lists Google-first validation dataset candidates.
- `scripts/export_google_fleurs_fixtures.py` can export local D1/D2 fixture wavs from `google/fleurs`, including a direct repo fallback for older `datasets` versions.
- `scripts/build_long_fixture.py` can concatenate existing fixture wavs into a local D3 long-form validation input.
- `scripts/build_diarization_fixture.py` can derive a local D4 synthetic mixture from existing mono fixtures.
- `scripts/build_failure_fixtures.py` can derive a local D5 failure-injection set from an existing wav.
- `scripts/build_ja_reading_map.py` can generate a starter kanji-reading map for Japanese alignment.
- `scripts/build_alignment_text_map.py` can generate a starter generic token map for non-latin alignment.
- `docs/VALIDATION_DATASET_MANIFEST.md` records the current local D1/D2/D3/D4/D5 fixture hashes and durations exported on 2026-03-21.
- `docs/BENCHMARK_TEMPLATE.md` records the current local smoke benchmark baseline and the command used to reproduce it.
- Validation, compatibility, and pending design choices are tracked in `IMPLEMENTATION_TASKS.md` and `DECISIONS.md`.
- Docker scaffolding is available through `Dockerfile`.
