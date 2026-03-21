# whisper-omega Implementation Tasks

最終更新: 2026-03-20

この文書は `whisper-omega-plan/` 配下の要件・仕様を実装タスクへ落とした作業台帳である。MVP 完成をゴールに、着手順・依存関係・受け入れ条件を整理する。

## Current Status

- 完了: T1, T2, T3, T4, T5, T6
- 実質完了: T11, T12, T13, T14, T20, T21
- 部分完了: T5, T7, T8, T9, T10, T15, T16, T17, T18, T19, T22
- 未着手: なし

補足:
- `faster-whisper` が未導入のため、ASR 実行は dependency failure を返す形で実装済み
- P1 / P2 別紙は v0.1 のたたき台を追加済み
- `faster-whisper` による CPU smoke test は通過済み
- `pyannote.audio` は導入済みで、`HF_TOKEN` 未設定時の configuration/degraded 経路まで確認済み
- alignment は既存 word timestamps を用いる best-effort 実装、diarization は token 設定で実 backend 実行可能

参照元:
- `whisper-omega-plan/requirements/whisper_omega_v05.md`
- `whisper-omega-plan/specs/appendix_a_schema_v01.md`
- `whisper-omega-plan/specs/appendix_b_compat_v01.md`
- `whisper-omega-plan/specs/appendix_c_cli_v01.md`
- `whisper-omega-plan/specs/appendix_d_validation_v01.md`

## 0. ゴール

MVP として以下を満たす。

- `omega transcribe` を正式 CLI として提供する
- JSON / SRT / VTT / TXT 出力を提供する
- JSON Schema `1.0.0` に準拠する
- `success` / `degraded` / `failure` と exit code family を実装する
- `runtime-policy` と `device` の挙動を仕様通り固定する
- `omega doctor` と `omega setup core` を提供する
- WhisperX v3.8.1 互換対象の CLI / JSON / exit family テストを通す
- 第一級サポート環境向けの検証手順と測定基盤を揃える

## 1. すぐ着手するタスク

### T1. プロジェクト骨格を作る

目的:
- 実装・テスト・将来の backend 拡張がやりやすい最小構成を作る

完了条件:
- Python プロジェクトとして初期化されている
- `src/` と `tests/` がある
- パッケージ名、CLI エントリポイント、依存管理方針が決まっている
- README の土台がある

想定構成:
- `src/whisper_omega/cli/`
- `src/whisper_omega/runtime/`
- `src/whisper_omega/io/`
- `src/whisper_omega/asr/`
- `src/whisper_omega/align/`
- `src/whisper_omega/diarize/`
- `src/whisper_omega/vad/`
- `src/whisper_omega/compat/`
- `tests/unit/`
- `tests/integration/`
- `tests/e2e/`

### T2. 結果モデルと JSON 契約を実装する

目的:
- Appendix A の schema を実装の中心契約にする

完了条件:
- `schema_version=1.0.0` 固定
- トップレベル必須フィールドを持つ結果モデルがある
- `status` ごとの `error_code` / `error_category` / `backend_errors` 制約を検証できる
- `segments[].speaker` / `words[].speaker` / `speakers` の null / 空配列規則を強制できる
- 時刻と数値の丸め規則を共通化できている

備考:
- 実装では schema ファイルと Python 側モデルの二重管理ずれを防ぐ

### T3. CLI の最小実装を作る

目的:
- `omega transcribe` の基本挙動を固定する

完了条件:
- `omega transcribe INPUT [OPTIONS]` が動く
- `--runtime-policy`, `--device`, `--output-format`, `--emit-result-json`, `--write-failure-json` を受理する
- `stdout` は JSON 専用、`stderr` は人間向け要約という責務分離ができている
- `--emit-result-json=auto` の TTY / 非 TTY 挙動が実装されている
- exit code `0/10/20/30/31/40` が仕様通りに出る

### T4. runtime policy と failure model を実装する

目的:
- permissive / strict / strict-gpu の振る舞いをコード化する

完了条件:
- `permissive` は許容 가능한降格だけ継続する
- `strict` は要求未達時に failure へ落ちる
- `strict-gpu` は GPU 不可時に CPU 降格しない
- `--device=cuda` では permissive でも CPU 自動降格しない
- degraded 時に `fallbacks`, `failed_features`, `backend_errors` が埋まる

### T5. ASR only パイプラインを完成させる

目的:
- MVP の最小価値を先に成立させる

完了条件:
- faster-whisper backend で音声入力からテキストと segment を得られる
- word timestamps を扱える
- JSON / TXT / SRT / VTT 出力までつながる
- model cache と基本設定が使える
- 最低限の音声 decode エラーハンドリングがある

## 2. その次に進めるタスク

### T6. backend 抽象化を入れる

目的:
- ASR / alignment / diarization / VAD を交換可能にする

完了条件:
- `ASRBackend`, `AlignmentBackend`, `DiarizationBackend`, `VADBackend` の抽象境界がある
- 実装ごとの差分を runtime 層から隠せる
- backend 失敗を canonical error に正規化できる

### T7. alignment を統合する

目的:
- optional feature として alignment を扱えるようにする

完了条件:
- `--require-alignment` と `--require-feature alignment` が等価
- `--align-backend {none,wav2vec2}` を扱える
- permissive 時は degraded で継続、strict 時は failure にできる
- `words[]` を alignment 有無で一貫した契約で返せる

### T8. diarization を統合する

目的:
- plugin / backend として diarization を分離したまま提供する

完了条件:
- `--require-diarization` と `--require-feature diarization` が等価
- `--diarize-backend {none,pyannote}` を扱える
- 未実行時でも `speaker` フィールド省略が起きない
- `speakers=[]` / `speaker=null` 規則が常に守られる

### T9. WhisperX 互換フロントを作る

目的:
- WhisperX 利用者が移行しやすい受け口を用意する

完了条件:
- `omega whisperx` が動く
- 互換対象オプションの受理と内部変換ができる
- `--batch_size`, `--output_format`, `--highlight_words` などの alias を吸収できる
- Appendix B の CLI-001〜009 を通す

### T10. Python API の最小版を作る

目的:
- CLI 以外の組み込み利用を可能にする

完了条件:
- entrypoint がある
- `runtime_policy`, `required_features`, `device` を意味論互換で扱える
- CLI と同じ結果モデルを返せる
- 例外と failure result の境界が文書化されている

## 3. 運用・導入まわりのタスク

### T11. `omega doctor` を実装する

目的:
- 導入失敗の切り分けを早くする

完了条件:
- Python / OS / device / CUDA 系の基本診断ができる
- known error code へ分類できる
- 既知セットアップ失敗の 90% 分類目標に向けた出力形式を持つ

### T12. `omega setup core` を実装する

目的:
- documented path でのセットアップ成功率を上げる

完了条件:
- コア利用に必要な導線が一つにまとまっている
- doctor と組み合わせた復旧導線を案内できる
- README の最短手順に載せられる

### T13. Docker 導線を整える

目的:
- 第一級サポート環境で再現性の高い導線を持つ

完了条件:
- Linux x86_64 向け Docker イメージまたはビルド手順がある
- README に最短起動例がある
- doctor / transcribe の基本動作をコンテナ内で確認できる

### T14. README と運用文書を整える

目的:
- 利用者が推理せず始められる状態にする

完了条件:
- 最短転写手順がある
- `omega doctor` と `omega setup core` の案内がある
- strict / permissive / strict-gpu の違いを説明できている
- 互換フロントの位置づけが書かれている

## 4. テスト・検証タスク

### T15. schema / unit test を作る

目的:
- 契約破壊を早期検出する

完了条件:
- JSON schema 検証テストがある
- 丸め、null、空配列、status ごとの制約テストがある
- canonical error 変換テストがある

### T16. CLI / e2e test を作る

目的:
- 出力・終了コード・TTY 挙動を固定する

完了条件:
- `stdout` / `stderr` / ファイル出力の組み合わせを検証できる
- `--emit-result-json` の各モードを検証できる
- usage error 条件を検証できる
- exit code family を検証できる

### T17. 互換試験を作る

目的:
- WhisperX 互換対象を形式知として固定する

完了条件:
- CLI-001〜009 を機械実行できる
- JSON-001〜008 を機械実行できる
- EXIT-001〜006 を機械実行できる

### T18. validation データセット定義を埋める

目的:
- Appendix D の検証手順を実運用可能にする

完了条件:
- D1〜D5 の実ファイル候補が決まる
- SHA256、長さ、言語、話者数、expected notes を記録できる
- failure injection ケースの再現方法を明記できる

### T19. benchmark / 診断試験を作る

目的:
- 受け入れ判定に必要な性能・安定性・診断 KPI を測れるようにする

完了条件:
- `setup`, `cold_start`, `steady_state` を分けて測れる
- 各ケース 5 回、median 採用の仕組みがある
- WhisperX / faster-whisper 比較の記録が取れる
- 60 分バッチと failure injection を実施できる

## 5. 仕様保留のまま進めるタスク

以下は実装を止めず、設定値または TODO として扱う。

- `language` を短縮コードのまま許すか BCP-47 に厳密化するか
- `--diarize` を正式 alias にするか
- `--hf_token` の優先順位を CLI 優先にするか環境変数優先にするか
- failure 時 JSON の既定ファイル名を持たせるか
- CTranslate2 の固定 patch バージョン
- validation データセット D1〜D4 の実体

方針:
- まずは Whisper 系短縮コード許容で実装する
- `--diarize` は互換 alias として一旦受理する
- failure JSON の自動ファイル出力は無効を既定とする
- 未確定項目は `DECISIONS.md` または issue 化で追跡する

## 6. 受け入れ直前に必須のタスク

### T20. P1 別紙を追加する

完了条件:
- Appendix E: Python API 契約
- Appendix F: backend interface 定義
- Appendix G: SRT / VTT 出力詳細仕様

### T21. P2 別紙を追加する

完了条件:
- Appendix H: 監視・運用メトリクス仕様
- Appendix I: リリース / 互換性維持ポリシー

### T22. 受け入れ判定リハーサルを行う

完了条件:
- MUST 要件チェックリストが埋まる
- 第一級サポート環境で KPI を確認できる
- 互換試験、性能試験、診断試験の結果を保存できる

## 7. 推奨実装順

1. T1 プロジェクト骨格
2. T2 結果モデルと JSON 契約
3. T3 CLI 最小実装
4. T4 runtime policy と failure model
5. T5 ASR only パイプライン
6. T15/T16 単体・CLI テスト
7. T6 backend 抽象化
8. T7 alignment
9. T8 diarization
10. T9 WhisperX 互換フロント
11. T10 Python API
12. T11/T12 doctor と setup
13. T13/T14 Docker と README
14. T17〜T19 検証基盤
15. T20〜T22 受け入れ仕上げ

## 8. 現時点の着手判断

今すぐ始める実装対象:
- T1
- T2
- T3
- T4
- T5
- T15
- T16

理由:
- 仕様未確定点を抱えたままでも前進できる
- MVP の心臓部である CLI、結果契約、ASR only、failure model を先に固定できる
- 後続の alignment / diarization / 互換フロントが差し込みやすくなる
