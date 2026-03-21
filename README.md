# whisper-omega

`whisper-omega` is a new implementation aiming for a WhisperX-compatible UX with clearer runtime policy, structured failures, and stable JSON output contracts.

## Current MVP Scope

- `omega transcribe`
- `omega whisperx`
- `omega doctor`
- `omega setup core`
- JSON / SRT / VTT / TXT writers
- `success` / `degraded` / `failure` result model
- runtime policy: `permissive`, `strict`, `strict-gpu`
- optional `faster-whisper` integration when installed
- optional `torchaudio` forced alignment integration when installed
- optional `pyannote.audio` diarization integration when installed and configured
- WhisperX compatibility mapping for `--diarize`, `--align_model`, `--batch_size`, `--output_format`, and `--hf_token`

## Quick Start

```bash
python3 -m pip install -e .
omega doctor
omega setup core
omega setup align
omega setup diarize
omega setup validation
omega transcribe sample.wav --output-format json --emit-result-json always
```

If you are in an offline or sandboxed environment, package installation may require a local virtual environment plus `--no-build-isolation`.

```bash
python3 -m venv --system-site-packages .venv-system
.venv-system/bin/python -m pip install -e . --no-build-isolation
```

Optional extras:

```bash
.venv-system/bin/python -m pip install '.[core]' --no-build-isolation
.venv-system/bin/python -m pip install '.[align]' --no-build-isolation
.venv-system/bin/python -m pip install '.[diarize]' --no-build-isolation
.venv-system/bin/python -m pip install '.[validation]' --no-build-isolation
```

For pyannote diarization, set `HF_TOKEN` before running `omega transcribe --require-diarization --diarize-backend pyannote ...`. The backend prefers in-memory waveform loading via `torchaudio` when available, which helps avoid some decode-stack issues. You can also provide `OMEGA_PYANNOTE_NUM_SPEAKERS`, `OMEGA_PYANNOTE_MIN_SPEAKERS`, or `OMEGA_PYANNOTE_MAX_SPEAKERS` to constrain the diarization search space.
The `omega whisperx` compatibility frontend also accepts `--hf_token` and maps `--align_model` to alignment-required execution.
For forced alignment, install the `align` extra and run `omega transcribe --require-alignment --align-backend wav2vec2 ...`.
Current alignment coverage includes latin-script languages plus kana-only Japanese; other unsupported transcripts return a machine-readable alignment validation failure instead of silently degrading.
If you have an external romanizer, set `OMEGA_ALIGNMENT_ROMANIZER` and other non-latin transcripts can be pre-romanized before alignment.

## Notes

- Real transcription requires the optional `faster-whisper` dependency.
- Without it, `omega transcribe` returns a machine-readable dependency failure instead of crashing.
- `omega doctor` now reports known issue codes and recommended actions for missing runtime pieces.
- `scripts/generate_validation_report.py` can capture `doctor`, unit-test, and smoke results into one JSON report.
- `docs/VALIDATION_DATASET_CANDIDATES.md` lists Google-first validation dataset candidates.
- `scripts/export_google_fleurs_fixtures.py` can export local D1/D2 fixture wavs from `google/fleurs`, including a direct repo fallback for older `datasets` versions.
- `scripts/build_long_fixture.py` can concatenate existing fixture wavs into a local D3 long-form validation input.
- `scripts/build_diarization_fixture.py` can derive a local D4 synthetic mixture from existing mono fixtures.
- `scripts/build_failure_fixtures.py` can derive a local D5 failure-injection set from an existing wav.
- `docs/VALIDATION_DATASET_MANIFEST.md` records the current local D1/D2/D3/D4/D5 fixture hashes and durations exported on 2026-03-21.
- Validation, compatibility, and pending design choices are tracked in `IMPLEMENTATION_TASKS.md` and `DECISIONS.md`.
- Docker scaffolding is available through `Dockerfile`.
