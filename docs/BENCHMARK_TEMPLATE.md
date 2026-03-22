# Benchmark Template

最終更新: 2026-03-22

## Fixed Environment

- Python: 3.12.3
- OS: Linux-6.6.87.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
- GPU: NVIDIA GeForce RTX 3090
- Driver: 595.79
- CUDA: available via `torch.cuda`; local smoke GPU path currently fails before successful decode
- GPU acceptance command: `PYTHONPATH=src ./.venv-system/bin/python scripts/run_gpu_acceptance.py`
- Model: `tiny`
- Backend: `faster-whisper`
- Benchmark command: `python3 scripts/benchmark_smoke.py tmp_smoke.wav --model tiny --repeats 2 --devices cpu cuda`

## Case Matrix

| Case | Dataset | Mode | Run1 | Run2 | Run3 | Run4 | Run5 | Median | Mean | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ASR_ONLY | `tmp_smoke.wav` CPU | steady_state-ish | 15.5315 | 0.9940 |  |  |  | 8.2627 | 8.2627 | First run includes model warm-up; both runs `success` |
| ASR_ONLY | `tmp_smoke.wav` CUDA | steady_state-ish | 0.0311 | 0.0010 |  |  |  | 0.0160 | 0.0160 | Both runs `failure`; `error_code=AUDIO_DECODE_FAILURE` |
| ASR_ALIGNMENT | pending | steady_state |  |  |  |  |  |  |  | Use `--require-alignment --align-backend wav2vec2` after choosing fixed fixture |
| ASR_DIARIZATION | pending | steady_state |  |  |  |  |  |  |  | Requires `HF_TOKEN` and fixed pyannote setup |
| ASR_ALIGNMENT_DIARIZATION | pending | steady_state |  |  |  |  |  |  |  | Run after alignment+diarization acceptance conditions are fixed |

## Acceptance Summary

- WhisperX ratio: not yet measured
- faster-whisper ratio: local baseline only; direct comparison not yet measured
- 60-minute batch success rate: not yet measured
- strict-gpu fallback violations: none observed in unit/contract tests or `scripts/run_gpu_acceptance.py`
- GPU acceptance snapshot:
  - `device=auto` -> `actual_device=cuda`, residual `AUDIO_DECODE_FAILURE`
  - `device=cuda` -> `actual_device=cuda`, residual `AUDIO_DECODE_FAILURE`
  - `strict-gpu + auto` -> `actual_device=cuda`, residual `AUDIO_DECODE_FAILURE`
- Interpretation:
  - current GPU risk is decode-stack stability, not silent fallback away from CUDA
