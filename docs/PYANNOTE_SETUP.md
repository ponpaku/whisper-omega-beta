# Pyannote Setup Guide

この文書は `whisper-omega` で `pyannote` diarization を使うための実務向けメモです。

## 1. 必要なもの

- `.[diarize]` extra の導入
- Hugging Face アカウント
- `HF_TOKEN`
- Hugging Face 側で gated model の利用条件承認

推奨構成:

- `pyannote.audio`
- `torchaudio`
- `HF_TOKEN`

`torchaudio` が使える場合は in-memory waveform 経路を優先するため、decode 周りで詰まりにくくなります。

## 2. インストール

```bash
./.venv-system/bin/python -m pip install '.[diarize]' --no-build-isolation
```

必要に応じて `doctor` で状態を確認します。

```bash
PYTHONPATH=src ./.venv-system/bin/python -m whisper_omega doctor --json-output
```

最低限見たい項目:

- `hf_token_configured`
- `pyannote_ready`
- `backend_statuses.diarization.pyannote.ready`
- `backend_statuses.decode.torchaudio.ready`

## 3. HF_TOKEN を用意する

Hugging Face の token page:

- <https://hf.co/settings/tokens>

推奨:

- Read token
- `Read access to contents of all public gated repos you can access` を有効化

シェルでは次のように設定します。

```bash
export HF_TOKEN=hf_xxx
```

このあと `doctor` を再実行して `hf_token_configured: true` になることを確認します。

## 4. 利用条件を承認する

`pyannote` は token を入れただけでは足りず、Hugging Face 側で gated model の利用条件承認が必要になることがあります。

今回の実測で承認が必要だったモデル:

- <https://hf.co/pyannote/speaker-diarization-3.1>
- <https://hf.co/pyannote/segmentation-3.0>
- <https://hf.co/pyannote/speaker-diarization-community-1>

各ページでログインした状態で `Agree and access` または同等のボタンを押してください。

## 5. 動作確認

まず acceptance:

```bash
HF_TOKEN=hf_xxx PYTHONPATH=src ./.venv-system/bin/python scripts/run_pyannote_acceptance.py
```

期待:

- `pyannote_missing_token` は token なしケースの expected failure
- `pyannote_with_token_and_hints` は `status=success`
- 全体として `all_passed=true`

次に実運用コマンド:

```bash
export HF_TOKEN=hf_xxx
PYTHONPATH=src ./.venv-system/bin/omega transcribe sample.wav \
  --device auto \
  --model small \
  --language ja \
  --require-diarization \
  --diarize-backend pyannote \
  --output-format json \
  --emit-result-json always
```

必要なら話者数ヒントも付けられます。

```bash
export OMEGA_PYANNOTE_MIN_SPEAKERS=1
export OMEGA_PYANNOTE_MAX_SPEAKERS=2
```

## 6. よくある詰まりどころ

### `HF_TOKEN_MISSING`

原因:

- `HF_TOKEN` が設定されていない
- `doctor` 実行時に環境変数が渡っていない

確認:

- `doctor` で `hf_token_configured: true` か

### `DIARIZATION_AUTH_FAILURE`

原因候補:

- gated model の利用条件未承認
- token に gated repo への read 権限がない

確認:

- Hugging Face の token 設定で gated repo access が有効か
- 上記 3 モデルのページで利用条件を承認済みか

### `Temporary failure in name resolution`

原因:

- Hugging Face へのネットワーク到達性不足

確認:

- 実行環境から `hf.co` / `huggingface.co` に到達できるか
- sandbox や firewall でブロックされていないか

### `torchcodec_importable: false`

これは主に `ffmpeg` + `torchcodec` 経路の問題です。`torchaudio` 経路が ready なら、`pyannote` 実行自体は通る場合があります。

まずは `doctor` で `backend_statuses.decode.torchaudio.ready: true` を確認してください。

### 実ファイルで `speakers must not contain duplicate ids`

これは過去にあった `pyannote` 話者集約の不具合で、現行 main では修正済みです。古い checkout を使っている場合は最新 main に更新してください。

## 7. 期待する着地

正常系では次が揃います。

- `doctor` で `pyannote_ready: true`
- `scripts/run_pyannote_acceptance.py` が green
- `omega transcribe --diarize-backend pyannote ...` が `status: "success"`
- 結果 JSON に `segments[].speaker` と `speakers[]` が入る
