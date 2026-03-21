# Appendix E: Python API 契約 v0.1

## 1. 目的

`whisper-Ω` の Python API entrypoint、入力契約、出力契約、失敗時の意味論を固定する。

## 2. MVP entrypoint

MVP の主要 entrypoint は以下とする。

```python
from whisper_omega import PolicyConfig, ServiceConfig, TranscriptionService, transcribe_file
```

最小利用例:

```python
config = ServiceConfig(
    policy=PolicyConfig(runtime_policy="permissive", device="cpu"),
    model_name="small",
)
service = TranscriptionService(config)
result = service.transcribe(Path("sample.wav"))
```

または facade:

```python
result = transcribe_file("sample.wav", device="cpu", runtime_policy="permissive")
```

## 3. 入力契約

最低限以下を意味論互換で扱う。

- `runtime_policy`
- `device`
- `required_features`
- `align_backend`
- `diarize_backend`
- `model_name`
- `language`

## 4. 出力契約

返却値は `TranscriptionResult` とし、JSON 出力時と同じ意味論を持つ。

- `schema_version`
- `status`
- `text`
- `language`
- `segments`
- `words`
- `speakers`
- `metadata`
- `error_code`
- `error_category`
- `backend_errors`

## 5. 失敗モデル

- usage error 相当は API 利用者側の入力検証エラーとする
- backend / runtime / dependency 起因の失敗は `TranscriptionResult(status="failure")` で返す
- permissive での部分機能欠落は `status="degraded"` で返す

## 6. 今後の拡張

- 高水準 `transcribe()` facade
- dataclass からの安定 import path 固定
- 例外ポリシーの明文化
