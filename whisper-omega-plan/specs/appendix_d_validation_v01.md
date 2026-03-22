# Appendix D: Validation Protocol v0.1

## 1. 目的
本書は受け入れ試験の再現手順を固定する。親要件文書 v0.5 の Appendix D に対応する。

## 2. 固定ベースライン
- WhisperX: v3.8.1（2026-02-14 release）citeturn346211search3turn346211search2
- faster-whisper: v1.2.1 citeturn346211search1turn346211search4
- CTranslate2: `4.5.x` に固定（正式 PATCH は実装開始時に確定）
- Python: 3.10.14
- Ubuntu: 22.04.4 LTS
- NVIDIA driver: 550.54.14
- CUDA: 12.x

## 3. 検証母集団
### 3.1 CPU
- x86_64, 16 vCPU 以上, RAM 32 GiB 以上

### 3.2 GPU
最低条件:
- compute capability 7.0 以上
- VRAM 8 GiB 以上

MVP 検証対象の例:
- RTX 3080 10GB
- RTX 4090 24GB
- L4 24GB

## 4. データセット固定
MVP 正式版では以下を固定する。
- `D1_SHORT_JA`: 日本語短尺音声 5 本
- `D2_SHORT_EN`: 英語短尺音声 5 本
- `D3_LONG_MIXED`: local provisional long-form 2 本
- `D4_DIARIZATION`: local synthetic diarization mix 3 本
- `D5_FAILURE_INJECTION`: local failure injection 6 ケース

### 4.1 実装前に埋める固定情報
- 各データセットのファイル名
- SHA256
- 長さ
- 言語
- 話者数
- expected notes

補助導線:
- `docs/DATASET_MANIFEST_TEMPLATE.md`
- `docs/VALIDATION_DATASET_MANIFEST.md`
- `docs/BENCHMARK_TEMPLATE.md`
- `docs/VALIDATION_DATASET_CANDIDATES.md`
- `scripts/export_google_fleurs_fixtures.py`
- `scripts/build_long_fixture.py`
- `scripts/build_diarization_fixture.py`
- `scripts/build_failure_fixtures.py`
- `scripts/build_dataset_manifest.py`
- `scripts/generate_validation_report.py`

### 4.2 2026-03-21 時点の推奨候補
- `D1_SHORT_JA`: `google/fleurs` `ja_jp` 由来の短尺 fixture
- `D2_SHORT_EN`: `google/fleurs` `en_us` 由来の短尺 fixture
- `D4_DIARIZATION`: `google/fleurs` 由来の multi-speaker synthetic mixture
- `D5_FAILURE_INJECTION`: `speech_commands` または `google/fleurs` 由来 fixture
- `D3_LONG_MIXED`: 当面は `facebook/voxpopuli` を fallback 候補としつつ、local provisional fixture は `build_long_fixture.py` で連結生成する

### 4.3 2026-03-21 時点で固定済みの local fixture
- `D1_SHORT_JA`: `fixtures/d1_short_ja` に 5 file export 済み
- `D2_SHORT_EN`: `fixtures/d2_short_en` に 5 file export 済み
- `D3_LONG_MIXED`: `fixtures/d3_long_mixed/d3_concat_01.wav` / `d3_concat_02.wav` を local provisional long-form input として生成済み
- `D4_DIARIZATION`: `fixtures/d4_diarization/d4_mix_01.wav` / `d4_mix_overlap_01.wav` / `d4_mix_3spk_01.wav` を local synthetic mixture として生成済み
- `D5_FAILURE_INJECTION`: `fixtures/d5_failure_injection` に decode failure fixture を生成済み。manifest には `OUTPUT_PERMISSION_DENIED` / `DEPENDENCY_MISSING` の scenario fixture も含む
- SHA256 / duration は `docs/VALIDATION_DATASET_MANIFEST.md` に固定

### 4.4 acceptance での扱い
- 必須ケース:
  - `doctor --json-output`
  - full unittest
  - ASR smoke (`scripts/run_smoke.sh`)
  - alignment smoke (`scripts/run_alignment_smoke.py`)
  - diarization smoke (`scripts/run_diarization_smoke.py`)
- 参考ケース:
  - `pyannote` 実依存 acceptance (`scripts/run_pyannote_acceptance.py`)
  - `nemo` 実依存 acceptance (`scripts/run_nemo_acceptance.py`)
  - GPU 実機 acceptance (`scripts/run_gpu_acceptance.py`)
  - `D3_LONG_MIXED` の長尺ベンチマーク
  - `D4_DIARIZATION` の overlap 強めケース
  - `D5_FAILURE_INJECTION` の個別 failure replay
- 標準実行導線:
  - `./.venv-system/bin/python scripts/run_acceptance.py`
- 標準成果物:
  - `validation-report.json`

### 4.5 pyannote 実依存 acceptance
- 実行コマンド:
  - `./.venv-system/bin/python scripts/run_pyannote_acceptance.py`
- 固定ケース:
  - `HF_TOKEN` 未設定: `HF_TOKEN_MISSING` を返すこと
  - `HF_TOKEN` 設定済み + `OMEGA_PYANNOTE_MIN_SPEAKERS=1` + `OMEGA_PYANNOTE_MAX_SPEAKERS=2`: success を狙うこと
- block 条件:
  - `faster-whisper` 未導入
  - `pyannote.audio` 未導入
  - `torchaudio` と `ffmpeg+torchcodec` のどちらも readiness を満たさない
- block 時:
  - JSON に `blocked=true` と `blocked_reasons` を残す
  - exit code は `2`

### 4.6 nemo 実依存 acceptance
- 実行コマンド:
  - `./.venv-system/bin/python scripts/run_nemo_acceptance.py`
- 固定ケース:
  - `OMEGA_NEMO_CONFIG` を無効パスにした状態: `CONFIG_INVALID` を返すこと
  - `OMEGA_NEMO_NUM_SPEAKERS=1` + `OMEGA_NEMO_MAX_SPEAKERS=2`: success を狙うこと
- block 条件:
  - `faster-whisper` 未導入
  - `nemo` 未導入
- block 時:
  - JSON に `blocked=true` と `blocked_reasons` を残す
  - exit code は `2`

### 4.7 GPU 実機 acceptance
- 実行コマンド:
  - `./.venv-system/bin/python scripts/run_gpu_acceptance.py`
- 固定ケース:
  - `device=auto`: `actual_device=cuda` を返すこと
  - `device=cuda`: `GPU_UNAVAILABLE` に落ちず CUDA 経路へ入ること
  - `runtime_policy=strict-gpu` + `device=auto`: CPU へ silent fallback しないこと
- 残留リスクの扱い:
  - `AUDIO_DECODE_FAILURE` は GPU 経路へ入った上での residual risk として `residual_risks` に記録する
  - `GPU_UNAVAILABLE` は block もしくは失敗として分離する
- block 条件:
  - `faster-whisper` 未導入
  - CUDA 不可
- block 時:
  - JSON に `blocked=true` と `blocked_reasons` を残す
  - exit code は `2`

## 5. benchmark 種別
- `setup`: download/setup を含む
- `cold_start`: model download を除外し、初回ロードを含む
- `steady_state`: model preload 済み、download/初回初期化を除外

性能受け入れ判定には `steady_state` を使う。

## 6. 測定ルール
- 各ケース 5 回実施
- 採用値は median
- mean は参考値として保存
- 失敗 run は別集計し、成功 run を補完するために無制限再試行しない
- 同一ケースで 5 回中 2 回以上 failure の場合、そのケースは不合格

### 6.1 cache 規則
- `setup`: cold cache
- `cold_start`: model cache あり、process cold
- `steady_state`: model cache あり、process warm

### 6.2 含める/除外する時間
- model download 時間: `setup` のみ含む
- initial model load: `cold_start` のみ含む
- 出力ファイル書き込み時間: すべて含む
- 外部後処理時間: 除外

## 7. 機能別ケース
- `ASR_ONLY`
- `ASR_VAD`
- `ASR_ALIGNMENT`
- `ASR_DIARIZATION`
- `ASR_ALIGNMENT_DIARIZATION`

## 8. 故障注入ケース（最低限）
- `GPU_UNAVAILABLE`
- `CUDA_MISMATCH`
- `OOM`
- `DEPENDENCY_MISSING`
- `CONFIG_INVALID`
- `ALIGNMENT_MODEL_UNAVAILABLE`
- `DIARIZATION_BACKEND_UNAVAILABLE`
- `HF_TOKEN_MISSING`
- `OUTPUT_PERMISSION_DENIED`
- `INVALID_ARGUMENT_COMBINATION`
- `AUDIO_DECODE_FAILURE`
- `EMPTY_INPUT`

## 9. 受け入れ項目
### 9.1 性能
- WhisperX 比 1.20 倍以内
- faster-whisper 直利用比 1.35 倍以内

### 9.2 安定性
- 60 分音声バッチ処理の非異常終了率 99% 以上

### 9.3 診断
- `omega doctor` による既知原因コード分類率 90% 以上

### 9.4 runtime policy
- `strict` / `strict-gpu` で silent degrade しない

## 10. 異常原因コード語彙（ドラフト）
- `GPU_UNAVAILABLE`
- `CUDA_MISMATCH`
- `OOM`
- `DEPENDENCY_MISSING`
- `CONFIG_INVALID`
- `ALIGNMENT_MODEL_UNAVAILABLE`
- `ALIGNMENT_BACKEND_UNAVAILABLE`
- `ALIGNMENT_LANGUAGE_UNSUPPORTED`
- `ALIGNMENT_TEXT_UNAVAILABLE`
- `ALIGNMENT_TEXT_UNSUPPORTED`
- `ALIGNMENT_RUNTIME_FAILURE`
- `ALIGNMENT_TEXT_MAP_INVALID`
- `DIARIZATION_BACKEND_UNAVAILABLE`
- `DIARIZATION_DECODE_UNAVAILABLE`
- `HF_TOKEN_MISSING`
- `DIARIZATION_AUTH_FAILURE`
- `DIARIZATION_MODEL_UNAVAILABLE`
- `DIARIZATION_DECODE_FAILURE`
- `DIARIZATION_CHANNELS_UNAVAILABLE`
- `DIARIZATION_CHANNEL_AMBIGUOUS`
- `DIARIZATION_AUDIO_UNSUPPORTED`
- `NEMO_MODEL_UNAVAILABLE`
- `NEMO_RUNTIME_FAILURE`
- `NEMO_OUTPUT_MISSING`
- `OUTPUT_PERMISSION_DENIED`
- `INVALID_ARGUMENT_COMBINATION`
- `AUDIO_DECODE_FAILURE`
- `EMPTY_INPUT`

`omega doctor --json-output` はこの canonical 語彙を `canonical_issue_codes` として返し、現環境で該当したものだけを `known_issue_codes` として返す。

## 11. 未確定事項
1. CTranslate2 の patch をどこで固定するか
2. GPU 実機 acceptance をどの母集団で固定するか
3. failure injection を CI でどこまで自動化するか
