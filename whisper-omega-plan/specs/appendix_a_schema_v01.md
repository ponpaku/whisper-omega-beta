# Appendix A: JSON Schema 仕様書 v0.1

## 1. 目的
本書は `whisper-Ω` の外部 JSON 契約を固定する。親要件文書 v0.5 の Appendix A に対応し、実装・テスト・クライアント統合・互換判定で拘束力を持つ。

## 2. スキーマバージョン
- JSON Schema draft: 2020-12
- `schema_version`: `1.0.0` 固定
- 互換破壊を伴う変更は `schema_version` のメジャー更新でのみ許可する

## 3. トップレベル契約
必須フィールド:
- `schema_version`
- `status`
- `text`
- `language`
- `segments`
- `words`
- `speakers`
- `metadata`

条件付きフィールド:
- `error_code`
- `error_category`
- `backend_errors`

### 3.1 `status`
列挙値:
- `success`
- `degraded`
- `failure`

### 3.2 `error_*` 規則
- `success`: `error_code=null`, `error_category=null`, `backend_errors=[]`
- `degraded`: `error_code`, `error_category` は null / non-null の両方を許容
- `failure`: `error_code`, `error_category` は必須で non-null

## 4. speaker 契約
- `segments[].speaker` は常に存在する
- `words[].speaker` は `words[]` 要素が存在する場合に常に存在する
- diarization 未実行・未利用・不明時は `null`
- 空文字は禁止
- フィールド省略は禁止
- `speakers` は未実行時でも存在し、空配列 `[]` とする

## 5. metadata 契約
必須:
- `asr_backend`
- `align_backend`
- `diarization_backend`
- `device`
- `requested_device`
- `actual_device`
- `fallbacks`
- `requested_features`
- `completed_features`
- `failed_features`

任意:
- `alignment_strategy`
- `alignment_token_source`

feature 語彙:
- `asr`
- `vad`
- `alignment`
- `diarization`
- `subtitle_export`

## 6. 数値丸め規則
- 時刻 `start` / `end`: 秒単位の浮動小数。出力丸めは小数点以下 3 桁
- `confidence`: 0.0〜1.0。出力丸めは小数点以下 4 桁
- `avg_logprob`: 出力丸めは小数点以下 4 桁

## 7. 追加プロパティ
- トップレベルおよび定義済みネストオブジェクトは `additionalProperties=false`
- 将来拡張は `schema_version` 更新または新たな optional field の明示追加で行う

## 8. `backend_errors[]` 契約
各要素は以下を持つ。
- `backend`: backend 名
- `code`: canonical code
- `category`: `usage | dependency | configuration | runtime | backend | validation | internal`
- `message`: 人間可読メッセージ
- `retryable`: boolean

## 9. `fallbacks[]` 契約
各要素は以下を持つ。
- `type`: `device | backend | feature | quality`
- `from`
- `to`
- `reason`

## 10. 代表例

### 10.1 success
```json
{
  "schema_version": "1.0.0",
  "status": "success",
  "text": "こんにちは",
  "language": "ja",
  "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "こんにちは", "speaker": null}],
  "words": [{"text": "こんにちは", "start": 0.0, "end": 1.2, "speaker": null, "confidence": 0.9981}],
  "speakers": [],
  "metadata": {
    "asr_backend": "faster-whisper",
    "align_backend": "none",
    "diarization_backend": "none",
    "device": "cpu",
    "requested_device": "auto",
    "actual_device": "cpu",
    "fallbacks": [],
    "requested_features": ["asr"],
    "completed_features": ["asr"],
    "failed_features": []
  },
  "error_code": null,
  "error_category": null,
  "backend_errors": []
}
```

### 10.2 degraded
```json
{
  "schema_version": "1.0.0",
  "status": "degraded",
  "text": "hello world",
  "language": "en",
  "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world", "speaker": null}],
  "words": [],
  "speakers": [],
  "metadata": {
    "asr_backend": "faster-whisper",
    "align_backend": "wav2vec2",
    "diarization_backend": "pyannote",
    "device": "cuda",
    "requested_device": "cuda",
    "actual_device": "cuda",
    "fallbacks": [{"type": "feature", "from": "alignment", "to": "segment_only", "reason": "ALIGNMENT_MODEL_UNAVAILABLE"}],
    "requested_features": ["asr", "alignment"],
    "completed_features": ["asr"],
    "failed_features": ["alignment"]
  },
  "error_code": "ALIGNMENT_MODEL_UNAVAILABLE",
  "error_category": "backend",
  "backend_errors": [
    {
      "backend": "wav2vec2",
      "code": "ALIGNMENT_MODEL_UNAVAILABLE",
      "category": "backend",
      "message": "alignment model could not be loaded",
      "retryable": false
    }
  ]
}
```

### 10.3 failure
```json
{
  "schema_version": "1.0.0",
  "status": "failure",
  "text": "",
  "language": "",
  "segments": [],
  "words": [],
  "speakers": [],
  "metadata": {
    "asr_backend": "faster-whisper",
    "align_backend": "none",
    "diarization_backend": "none",
    "device": "cuda",
    "requested_device": "cuda",
    "actual_device": "cpu",
    "fallbacks": [],
    "requested_features": ["asr"],
    "completed_features": [],
    "failed_features": ["asr"]
  },
  "error_code": "GPU_UNAVAILABLE",
  "error_category": "runtime",
  "backend_errors": [
    {
      "backend": "faster-whisper",
      "code": "GPU_UNAVAILABLE",
      "category": "runtime",
      "message": "CUDA device not available",
      "retryable": true
    }
  ]
}
```

## 11. 未確定事項
次を確認したい。
1. `speaker` の型を将来 ID 固定にするか、ラベル文字列を許容し続けるか
2. `language` を BCP-47 に厳密化するか、Whisper 系の短い言語コードをそのまま許容するか
3. `confidence` 非対応 backend で `null` とする方針を正式確定してよいか

## 12. 機械可読版
- `appendix_a_schema_v01.json`
