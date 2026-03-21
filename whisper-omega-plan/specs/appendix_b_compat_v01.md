# Appendix B: 互換性マトリクス v0.1

## 1. 目的
WhisperX v3.8.1 を基準に、whisper-Ω の互換対象・非互換対象・差分理由・テストケースを定義する。親要件文書 v0.5 の Appendix B に対応する。

基準版は WhisperX v3.8.1 とする。citeturn346211search3turn346211search2

## 2. 判定語彙
- `同等`: 同じ意味論で利用可能
- `部分互換`: 名称または挙動に差分があるが移行可能
- `非互換`: MVP では保証しない

## 3. CLI オプション差分台帳

| WhisperX v3.8.1 | whisper-Ω | 判定 | 差分理由 | 移行注意 | テストID |
|---|---|---|---|---|---|
| `--model` | `--model` | 同等 | 同名 | なし | CLI-001 |
| `--device` | `--device` | 部分互換 | `auto` の exact behavior を固定 | `cuda` は GPU 要求として扱う | CLI-002 |
| `--language` | `--language` | 同等 | 同名 | なし | CLI-003 |
| `--batch_size` | `--batch-size` または互換受理 | 部分互換 | 内部命名差分 | 互換フロント利用推奨 | CLI-004 |
| `--output_format` | `--output-format` | 部分互換 | 命名規則差分 | ハイフン版へ移行 | CLI-005 |
| `--diarize` | `--require-diarization` + backend 設定 | 部分互換 | policy 連携を導入 | backend 未設定時の規則を要確認 | CLI-006 |
| `--highlight_words` | `--highlight-words` または互換受理 | 部分互換 | 命名規則差分 | 互換フロント利用推奨 | CLI-007 |
| `--hf_token` | 互換受理可 | 部分互換 | backend 設定へマップ | 環境変数優先の可能性あり | CLI-008 |
| `--align_model` | `--align-model` または backend config | 部分互換 | backend 抽象化のため | backend 名とモデル指定を分離 | CLI-009 |

## 4. JSON フィールド差分台帳

| フィールド | WhisperX | whisper-Ω | 判定 | 差分理由 | テストID |
|---|---|---|---|---|---|
| `segments[]` | あり | あり | 同等 | 主要意味論維持 | JSON-001 |
| `words[]` | あり | あり | 同等 | 主要意味論維持 | JSON-002 |
| `segments[].speaker` | 実装依存 | 常時存在、未実行時 null | 部分互換 | 契約固定のため | JSON-003 |
| `words[].speaker` | 実装依存 | 常時存在、未実行時 null | 部分互換 | 契約固定のため | JSON-004 |
| `speakers[]` | 実装依存 | 常時存在、未実行時 [] | 部分互換 | 契約固定のため | JSON-005 |
| `metadata` | 非固定 | 必須 | 非互換寄りの拡張 | 可観測性強化 | JSON-006 |
| `status` | 非固定 | 必須 | 非互換寄りの拡張 | success/degraded/failure を固定 | JSON-007 |
| `schema_version` | 非固定 | 必須 | 非互換寄りの拡張 | 契約版管理のため | JSON-008 |

## 5. exit status family 対応
WhisperX 側は shell-level での成功/失敗を持つが、whisper-Ω は family を明示する。MVP では数値一致ではなく意味論一致を目標とする。

| whisper-Ω exit code | family | WhisperX との対応方針 | テストID |
|---|---|---|---|
| `0` | success | 正常終了と対応 | EXIT-001 |
| `10` | degraded success | WhisperX では明示 family がないため部分互換 | EXIT-002 |
| `20` | runtime failure | 実行時失敗と対応 | EXIT-003 |
| `30` | dependency failure | dependency 系失敗と対応 | EXIT-004 |
| `31` | configuration failure | config 系失敗と対応 | EXIT-005 |
| `40` | usage error | 引数不正と対応 | EXIT-006 |

## 6. 互換テストの最小集合
- CLI-001〜009
- JSON-001〜008
- EXIT-001〜006

これらは 100% 通過を MVP 条件とする。

## 7. 未確定事項
1. `--diarize` を `--require-diarization` の別名として正式採用するか
2. WhisperX 側の `--output_format` 命名を互換 alias としてどこまで維持するか
3. `metadata` 必須化を「非互換」ではなく「拡張互換」と表現するか
