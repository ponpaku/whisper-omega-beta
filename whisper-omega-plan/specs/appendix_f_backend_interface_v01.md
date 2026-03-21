# Appendix F: Backend Interface 定義 v0.1

## 1. 目的

ASR / alignment / diarization / VAD の backend 境界を固定し、本体と optional 実装を疎結合に保つ。

## 2. MVP 対象 interface

- `ASRBackend`
- `AlignmentBackend`
- `DiarizationBackend`
- `VADBackend`

## 3. 現在の実装状態

- `ASRBackend`: 実装あり
- `AlignmentBackend`: 仕様のみ
- `DiarizationBackend`: 仕様のみ
- `VADBackend`: 仕様のみ

## 4. ASRBackend 契約

入力:

- `audio_path`
- `model_name`
- `language`
- `device`
- `batch_size`

出力:

- `text`
- `language`
- `segments`
- `words`
- `backend_errors`

失敗時:

- dependency 起因は canonical code `DEPENDENCY_MISSING`
- runtime 起因は canonical code を本体側へ渡す

## 5. 将来 interface の期待

### AlignmentBackend

- 入力: segments, language, model/backend config
- 出力: word timings の補完結果

### DiarizationBackend

- 入力: audio_path, segments, backend config
- 出力: speaker labels と `speakers[]`

### VADBackend

- 入力: audio_path, backend config
- 出力: speech regions

## 6. 設計原則

- backend 例外をそのまま外へ漏らさない
- canonical error code へ正規化する
- unavailable backend は degraded / failure として機械可読に返す

