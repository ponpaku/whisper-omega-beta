# whisper-omega Implementation Tasks

最終更新: 2026-03-22

この文書は「完全完成」まで止まらず進めるための実行タスクリストである。  
`whisper-omega-plan/評価_202603211500.md` の評価内容と、現行実装の到達点を踏まえて更新している。

## Current Status

- 完了:
  - 結果モデル / JSON schema 契約の実装
  - `omega transcribe` / `omega whisperx` / `omega doctor` / `omega setup *` の基本導線
  - `runtime_policy` / `device` / exit family / failure model
  - `device=auto` の実デバイス選択
  - `strict-gpu` の failure 化
  - `doctor` の拡張診断
  - `wav2vec2` alignment の実行経路
  - `pyannote` diarization の実行経路
  - validation fixture / manifest / report 生成基盤
- 実質完了:
  - WhisperX 互換フロントの主要入口
  - validation の D1-D5 ローカル再現導線
  - non-latin alignment の補助フック
- 未完:
  - forced alignment の本格化
  - diarization の本番安定化
  - WhisperX 互換範囲の深掘り
  - acceptance / benchmark の固定

## 完成条件

以下を満たしたら「完全完成」とみなす。

1. `omega transcribe` が CPU / GPU の両方で安定実行できる
2. `success / degraded / failure` の JSON 契約が schema / fixture / CLI で一致する
3. forced alignment が英語以外を含む現実的な入力で実用になる
4. diarization が pyannote 実運用条件込みで安定完走できる
5. WhisperX 互換対象が文書・テスト・実装で一致する
6. validation protocol が固定データ / 固定手順 / 固定レポートまで閉じる
7. `doctor` が実運用の詰まりどころを十分に分類できる

## Milestone 1: Forced Alignment を完成域へ寄せる

### A1. alignment metadata を結果へ正式反映する
- 状態: 完了
- 目的: alignment が成功 / fallback / failure した理由を結果 JSON から追えるようにする
- すでにあるもの:
  - `metadata.alignment_strategy`
  - `metadata.alignment_token_source`
  - permissive / strict の結果で metadata を保持する service 実装
  - Appendix A schema 反映
  - representative JSON fixture 反映
- 完了条件:
  - metadata に alignment strategy / map usage / romanizer usage が出る
  - degraded / failure 時に alignment cause code が固定語彙で出る
  - representative JSON fixtures を追加する

### A2. non-latin alignment の一般拡張点を固定する
- 状態: 進行中
- すでにあるもの:
  - `OMEGA_ALIGNMENT_ROMANIZER`
  - `OMEGA_ALIGNMENT_JA_READING_MAP`
  - `OMEGA_ALIGNMENT_TEXT_MAP`
  - `scripts/build_ja_reading_map.py`
  - `scripts/build_alignment_text_map.py`
- 残タスク:
  - map / romanizer / built-in fallback の優先順位を docs とコードで固定
  - unsupported ケースの error code を整理
  - `doctor` に alignment strategy summary をさらに載せる

### A3. language/model 解決戦略を明文化する
- 状態: 完了
- 目的: 言語ごとに「何で alignment するか」を仕様化する
- 完了条件:
  - latin-script
  - Japanese kana
  - Japanese with reading map
  - generic text map
  - external romanizer
  - unsupported
  の分岐が文書化されている

### A4. forced alignment smoke test を実データに接続する
- 状態: 完了
- 完了条件:
  - D2_SHORT_EN に対する alignment smoke test を 1 本固定
  - D1_SHORT_JA に対する kana / reading-map 経路を 1 本固定
  - validation report に alignment smoke を載せられる

## Milestone 2: Diarization を本番寄りに安定化する

### D1. pyannote 実行条件を固定する
- 状態: 完了
- すでにあるもの:
  - `HF_TOKEN` 診断
  - `torchaudio` / `ffmpeg+torchcodec` decode path 診断
  - speaker hint env (`NUM/MIN/MAX`)
  - diarization runtime failure の auth / model / decode / config 分類
- 残タスク:
  - なし

### D2. diarization result 契約を強化する
- 状態: 完了
- 完了条件:
  - `speakers[]` / `segments[].speaker` / `words[].speaker` の整合性テストを追加
  - overlap や speaker assignment edge case の fixture を追加

### D3. D4 synthetic mixture を増やす
- 状態: 部分完了
- すでにあるもの:
  - `fixtures/d4_diarization/d4_mix_01.wav`
- 残タスク:
  - 2-speaker / 3-speaker / overlap 強めの 3 ケースへ拡張
  - recipe と manifest を更新

## Milestone 3: WhisperX 互換を完成域へ寄せる

### W1. 互換対象オプション一覧を固定する
- 状態: 未着手
- 完了条件:
  - supported / partial / unsupported を表で固定
  - Appendix B / README / CLI 実装が一致する

### W2. 非対応引数の扱いを固定する
- 状態: 未着手
- 完了条件:
  - silent ignore を減らす
  - warning / usage error / partial compatibility の基準を決める
  - test case を追加する

### W3. JSON / exit family 互換試験を増やす
- 状態: 部分完了
- 完了条件:
  - CLI-001〜009
  - JSON-001〜008
  - EXIT-001〜006
  の自動実行を揃える

## Milestone 4: Validation / Acceptance を固定する

### V1. D3/D4/D5 を正式 fixture として固める
- 状態: 進行中
- すでにあるもの:
  - D3 provisional long-form
  - D4 synthetic diarization
  - D5 failure injection
- 残タスク:
  - D3 を複数ケースへ増やす
  - D4 を複数ケースへ増やす
  - D5 に permission / missing dependency 系も追加する

### V2. benchmark template を埋める
- 状態: 未着手
- 完了条件:
  - CPU / GPU / setup / cold_start / steady_state の計測欄が埋まる
  - README から辿れる

### V3. validation report を smoke 付きで固定する
- 状態: 部分完了
- 完了条件:
  - doctor
  - full unittest
  - asr smoke
  - alignment smoke
  - diarization smoke
  を 1 つの report に集約できる

## Milestone 5: 契約と運用の仕上げ

### C1. representative JSON fixtures を追加する
- 状態: 未着手
- 完了条件:
  - success
  - degraded alignment
  - degraded diarization
  - hard failure
  の fixture があり schema validate を通る

### C2. Python API 契約を固定する
- 状態: 未着手
- 完了条件:
  - public entrypoint
  - example
  - failure / exception 境界
  - same result model
  を文書化する

### C3. release readiness を揃える
- 状態: 未着手
- 完了条件:
  - README の最短導線が最新
  - DECISIONS と IMPLEMENTATION_TASKS が最新
  - validation report が最新
  - 完成宣言に耐える変更履歴が揃う

## 実行順

1. A1
2. A3
3. A4
4. D1
5. D2
6. D3
7. W1
8. W2
9. W3
10. V1
11. V2
12. V3
13. C1
14. C2
15. C3

## 直近の次アクション

- D4 diarization fixture を 2 ケース以上に増やす
- WhisperX 互換対象表を Appendix B と README に反映する
