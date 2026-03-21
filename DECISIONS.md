# Decisions

## Temporary Decisions

- `language` currently accepts Whisper-style short codes.
- `--diarize` is accepted as a compatibility alias for `--require-diarization`.
- failure JSON is not written to a file unless `--write-failure-json` is explicitly set.
- `faster-whisper` is an optional dependency; if unavailable, the CLI returns a dependency-classified failure result.
- forced alignment currently targets latin-script languages via `torchaudio` `MMS_FA`; unsupported languages return `ALIGNMENT_LANGUAGE_UNSUPPORTED`.

## Open Decisions

- Whether to enforce BCP-47 for `language`
- Final precedence for `--hf_token` vs environment variables
- Final fixed CTranslate2 patch version
- Validation dataset file roster and hashes
