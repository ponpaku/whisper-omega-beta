# Appendix H: 監視・運用メトリクス仕様 v0.1

## 1. 目的

MVP 以降の運用監視で必要になるイベント、結果分類、診断軸を固定する。

## 2. 最低限の結果メトリクス

- `transcription_requests_total`
- `transcription_success_total`
- `transcription_degraded_total`
- `transcription_failure_total`
- `transcription_duration_seconds`

## 3. ラベル候補

- `asr_backend`
- `align_backend`
- `diarization_backend`
- `requested_device`
- `actual_device`
- `status`
- `error_category`
- `error_code`

## 4. doctor / setup 観測項目

- documented path completion
- known failure classification hit rate
- dependency availability snapshot

## 5. MVP との関係

- server mode health check は将来項目
- MVP では structured result と doctor output を一次情報源とする

