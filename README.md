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
- optional `pyannote.audio` diarization integration when installed and configured
- WhisperX compatibility mapping for `--diarize`, `--align_model`, `--batch_size`, `--output_format`, and `--hf_token`

## Quick Start

```bash
python3 -m pip install -e .
omega doctor
omega setup core
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
.venv-system/bin/python -m pip install '.[diarize]' --no-build-isolation
```

For pyannote diarization, set `HF_TOKEN` before running `omega transcribe --require-diarization --diarize-backend pyannote ...`.
The `omega whisperx` compatibility frontend also accepts `--hf_token` and maps `--align_model` to alignment-required execution.

## Notes

- Real transcription requires the optional `faster-whisper` dependency.
- Without it, `omega transcribe` returns a machine-readable dependency failure instead of crashing.
- Validation, compatibility, and pending design choices are tracked in `IMPLEMENTATION_TASKS.md` and `DECISIONS.md`.
- Docker scaffolding is available through `Dockerfile`.
