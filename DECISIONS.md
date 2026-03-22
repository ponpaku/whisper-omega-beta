# Decisions

## Temporary Decisions

- `language` currently accepts Whisper-style short codes.
- `--diarize` is accepted as a compatibility alias for `--require-diarization`.
- failure JSON is not written to a file unless `--write-failure-json` is explicitly set.
- `faster-whisper` is an optional dependency; if unavailable, the CLI returns a dependency-classified failure result.
- forced alignment currently targets latin-script languages via `torchaudio` `MMS_FA`; unsupported languages return `ALIGNMENT_LANGUAGE_UNSUPPORTED`.
- alignment token resolution currently follows this precedence: `OMEGA_ALIGNMENT_TEXT_MAP` -> native latin tokens / Japanese kana path -> `OMEGA_ALIGNMENT_JA_READING_MAP` -> `OMEGA_ALIGNMENT_ROMANIZER` -> unsupported.

## Open Decisions

- Whether to enforce BCP-47 for `language`
- Final precedence for `--hf_token` vs environment variables
- Final fixed CTranslate2 patch version
- Whether GPU `AUDIO_DECODE_FAILURE` should remain a documented residual risk or be promoted back to a release blocker after decode stack changes
