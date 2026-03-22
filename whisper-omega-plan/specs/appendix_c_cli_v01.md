# Appendix C: CLI 詳細仕様 v0.1

## 1. 目的
本書は CLI の exact behavior を固定する。親要件文書 v0.5 の Appendix C に対応し、実装・e2e テスト・互換判定で拘束力を持つ。

## 2. 対象コマンド
MVP 正式コマンド:
- `omega transcribe`

MVP 互換フロント:
- `omega whisperx` （WhisperX 互換フロント）
- `omega whisper` （OpenAI Whisper 互換フロント、任意）

## 3. 基本構文
```bash
omega transcribe INPUT [OPTIONS]
```

## 4. デフォルト
- CLI 単発実行の既定 `--runtime-policy`: `permissive`
- 既定 `--device`: `auto`
- 既定 `--emit-result-json`: `auto`

## 5. 主要オプション
- `--runtime-policy {permissive,strict,strict-gpu}`
- `--device {auto,cpu,cuda}`
- `--require-feature FEATURE` （複数回指定可）
- `--require-diarization`
- `--require-alignment`
- `--output-format {json,srt,vtt,txt}`
- `--output-file PATH`
- `--emit-result-json {auto,always,on-failure,never}`
- `--write-failure-json`
- `--diarize-backend {none,channel,nemo,pyannote}`
- `--align-backend {none,wav2vec2}`

### 5.1 糖衣構文
- `--require-diarization` ≡ `--require-feature diarization`
- `--require-alignment` ≡ `--require-feature alignment`

### 5.2 WhisperX 互換フロントでの既定 backend
- `omega whisperx --diarize` は `pyannote` backend を既定とする
- `channel` / `nemo` は `omega transcribe --diarize-backend ...` でのみ明示選択する

## 6. `--emit-result-json` exact behavior
### 6.1 `always`
- `success` / `degraded` / `failure` の全結果で JSON を `stdout` に出力する

### 6.2 `on-failure`
- `failure` 時のみ JSON を `stdout` に出力する
- `success` / `degraded` では JSON を `stdout` に出さない

### 6.3 `never`
- いかなる状態でも JSON を `stdout` に出力しない

### 6.4 `auto`
以下の規則を適用する。
- `stdout` が TTY の場合:
  - `success` / `degraded`: JSON を `stdout` に出さない
  - `failure`: JSON を `stdout` に出さない
- `stdout` が非 TTY の場合:
  - `success` / `degraded` / `failure` の全結果で JSON を `stdout` に出力する

この規則により、対話利用では可読性、パイプや自動処理では機械可読性を優先する。

## 7. stdout / stderr / file 出力
### 7.1 stdout
- 機械可読 JSON 専用
- 人間向けログを混ぜない

### 7.2 stderr
- 警告
- 進捗
- 人間向けエラー要約
- degraded / failure の簡潔な説明

### 7.3 ファイル出力
- `--output-file` 指定時、`success` / `degraded` の主出力を指定先へ書く
- `failure` 時に JSON をファイルへ書くのは `--write-failure-json` 指定時のみ
- `--write-failure-json` 指定なしでは、failure 時に主出力ファイルを生成しない

## 8. 競合・優先順位
優先順位は次の通り。
1. 明示 CLI 引数
2. 互換フロントによる変換
3. デフォルト値

### 8.1 usage error となる組み合わせ
- `--runtime-policy strict-gpu --device cpu`
- `--output-format txt --write-failure-json` と JSON 出力先未指定
- `--diarize-backend none --require-diarization`
- `--align-backend none --require-alignment`

## 9. `--device` と policy の意味
- `--device=cuda`: GPU を要求する。`permissive` でも CPU 自動降格しない
- `--device=cpu`: CPU 固定
- `--device=auto`: 利用可能な最適デバイスを選択。`permissive` では CPU 降格可、`strict-gpu` では不可

## 10. exit code
- `0`: success
- `10`: degraded
- `20`: runtime failure
- `30`: dependency failure
- `31`: configuration failure
- `40`: usage / argument error

## 11. 互換 CLI フロント範囲
MVP で互換対象とする WhisperX 主要オプション:
- `--model`
- `--device`
- `--language`
- `--batch_size`
- `--output_format`
- `--diarize`
- `--highlight_words`
- `--hf_token`（受理はするが、実体は backend 設定へマップ）
- `--align_model`

詳細な対応表は Appendix B に従う。

## 12. 代表シナリオ
### 12.1 対話利用
```bash
omega transcribe sample.wav --output-format srt
```
- `stdout`: JSON なし（TTY 前提）
- `stderr`: 進捗と要約
- exit code: 0 または 10 または 20/30/31/40
- ファイル: SRT

### 12.2 自動処理
```bash
omega transcribe sample.wav --output-format json | jq .
```
- `stdout`: JSON（非 TTY）
- `stderr`: ログ
- exit code: 上記と同じ

### 12.3 failure JSON を常に取りたい
```bash
omega transcribe sample.wav --emit-result-json always
```

## 13. 未確定事項
次を確認したい。
1. `--output-file` 未指定時の JSON 既定ファイル名を持たせるか
2. 互換フロントで `--hf_token` を直接受けるか、環境変数優先にするか
3. `--emit-result-json=auto` の TTY 判定を Windows でも同一規則にするか
