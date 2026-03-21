# whisper-Ω 要件定義書（ドラフト v0.5）

- 作成日: 2026-03-12
- 文書状態: Draft for Review
- 想定読者: 企画者、実装者、レビュー担当、将来のコントリビューター
- 本書の位置づけ: MVP 受け入れ判定の基準となる親要件文書
- 正式凍結条件: 本書に加え、指定の別紙仕様と検証手順が承認済みであること

---

## 1. 文書の目的

本書は、`whisper-Ω` の MVP（最小実用版）に関する要件、受け入れ判定基準、サポート範囲、互換範囲、および正式版凍結に必要な補助仕様の境界を定義する。  
`whisper-Ω` は、WhisperX の延命版ではなく、**WhisperX の上位互換 UX を目指す新規実装**として位置付ける。

本書の目的は以下の通り。

- プロダクトの狙いを明確化する
- WhisperX の既知課題に対する改善方針を整理する
- MVP の機能範囲を定義する
- 受け入れ判定基準を明文化する
- サポート対象環境と互換保証範囲を固定する
- 正式版凍結前に必要な別紙仕様と優先順位を固定する

---

## 2. 文書体系

### 2.1 親文書と別紙の関係

本書は親要件文書とする。  
実装解釈のブレを防ぐため、以下の別紙を正式仕様パッケージの一部として扱う。

### 2.2 別紙の優先順位

正式版凍結前に整備すべき別紙を、以下の優先順位で定義する。

#### P0: 実装着手前に固定が必要

1. **Appendix A: JSON Schema 本体**
2. **Appendix B: 互換性マトリクス（WhisperX v3.8.1 基準）**
3. **Appendix C: CLI 詳細仕様（runtime policy / require-feature 含む）**
4. **Appendix D: 検証手順書（Validation Protocol、ベースライン固定情報含む）**

#### P1: MVP 実装並行で固定

5. **Appendix E: Python API 契約**
6. **Appendix F: backend interface 定義**
7. **Appendix G: SRT / VTT 出力詳細仕様**

#### P2: MVP 受け入れ前までに整備

8. **Appendix H: 監視・運用メトリクス仕様（server mode health check 含む）**
9. **Appendix I: リリース / 互換性維持ポリシー**

### 2.3 正式版凍結の条件

以下を満たさない限り、本書単体では正式凍結としない。

- P0 別紙がすべて承認済みであること
- Appendix A の `schema_version=1.0.0` が固定済みであること
- Appendix B に差分台帳が記載済みであること
- Appendix D に測定手順・再試行条件・統計処理が定義済みであること

---

## 3. 背景

WhisperX は以下の機能を統合して提供することで実務価値を持つ。

- 高速 ASR
- 単語単位タイムスタンプ
- forced alignment
- VAD
- speaker diarization
- 字幕出力

一方で、現行 WhisperX には以下の系統の課題がある。

- 依存関係の結合が強い
- GPU / CUDA / cuDNN / CTranslate2 周辺で環境差の影響を受けやすい
- diarization が特定 backend に強く結びついている
- インストール成功と実行成功の間にギャップがある
- 本番運用時の可観測性・故障切り分け性が弱い

`whisper-Ω` は、機能面では WhisperX と同等以上を目指しつつ、**導入性、安定性、可観測性、運用性**を改善することを目的とする。

---

## 4. プロダクト定義

### 4.1 位置づけ

`whisper-Ω` は以下の性質を持つ。

- WhisperX 互換 UX を持つ新規実装
- faster-whisper を中核に据えた実務向け転写基盤
- alignment / diarization を交換可能 backend として統合するオーケストレータ

### 4.2 目標

- WhisperX ユーザーが移行可能であること
- faster-whisper 直利用者に対して完成品として優位性を持つこと
- 導入時にユーザーが依存関係を推理しなくてよいこと
- 本番運用で degraded / failure / backend 状態を把握できること

### 4.3 非目標

初期段階では以下を目標にしない。

- `pip install` 一発で全機能・全環境・全 GPU 世代を完全自動化すること
- WhisperX の内部実装互換を維持すること
- 初版から全 OS / 全アーキテクチャを第一級サポートすること
- 独自 ASR モデル研究そのものを主目的にすること

---

## 5. 想定ユースケース

### 5.1 WhisperX からの移行

- 既存の WhisperX 利用者が、同等以上の機能をより安定して利用する
- WhisperX 互換 CLI / 出力形式を活かして置き換える

### 5.2 faster-whisper 直利用からの移行

- faster-whisper を直接呼び出して glue code を書いている利用者が、alignment / diarization / subtitle export をまとめて利用する

### 5.3 製品・サービスへの組み込み

- バッチ処理
- API サーバー組み込み
- 会議録生成
- 字幕生成
- 監視可能な本番運用

---

## 6. WhisperX 比の主要改善点

- 本体依存の最小化
- diarization backend の分離・交換可能化
- `doctor` / `setup` / Docker を正式導線化
- 可観測性の強化
- strict / permissive の runtime policy 導入
- degraded / failure の区別明確化
- backend 障害の局所化
- 再現可能な配布・環境管理

---

## 7. WhisperX の課題に対する整理

### 7.1 現設計で対処方針を定義済みのもの

- 依存関係の密結合
- pyannote 固定の diarization 構造
- インストール成功と実行成功のギャップ
- backend 障害が全体障害になりやすい構造
- 可観測性不足

### 7.2 対処可能だが完全には消せないもの

- CUDA / cuDNN / CTranslate2 の相性問題
- 上流依存更新の影響
- 言語別 alignment モデル事情
- 一部プラットフォームでの導入の難しさ

### 7.3 対処不可能なもの

- 外部モデル配布・認証・利用条件
- NVIDIA ランタイムの実環境依存
- 上流プロジェクトの破壊的変更そのもの
- 未整備言語向け資産が存在しない問題

### 7.4 対処不要と判断するもの

- 真の意味での完全 one-command universal install
- WhisperX の内部実装互換
- 初版からの全環境完全対応

---

## 8. サポート対象環境

### 8.1 第一級サポート対象（MVP）

MVP の正式サポート対象は以下とする。

- Python: 3.10, 3.11
- OS: Ubuntu 22.04 LTS x86_64
- CPU: x86_64 CPU 実行
- GPU: NVIDIA GPU
- CUDA 系統: CUDA 12 系
- コンテナ: Docker on Linux x86_64
- 最低 NVIDIA driver: 550.54.14 以上
- 最低 compute capability: 7.0 以上
- 最低 GPU VRAM: 8 GiB 以上

上記の driver / compute capability / VRAM を満たさない GPU は、MVP の第一級サポート対象に含めない。  
Appendix D には、検証母集団として用いる GPU 型番一覧と driver バージョンを記載する。

### 8.2 ベストエフォート対象

以下は CI または限定検証の対象とし、正式保証は行わない。

- Python 3.12
- Windows 11 x86_64
- macOS 14+ Apple Silicon（CPU 実行のみ）
- Linux aarch64

### 8.3 非対応（MVP）

以下は MVP の対象外とする。

- CUDA 11 系での正式保証
- ROCm / AMD GPU
- Apple GPU 最適化
- 複数 GPU 分散実行
- リアルタイム配信向け低遅延保証

---

## 9. コア機能要件

### 9.1 MVP 必須機能

- faster-whisper ベース ASR
- word timestamps
- JSON / SRT / VTT / TXT 出力
- WhisperX 互換に近い出力意味論
- CLI 提供
- Python API 提供
- model cache 管理
- `omega doctor`
- `omega setup core`
- runtime policy 制御

### 9.2 optional 機能

- forced alignment
- diarization
- 高度 VAD backend
- Docker image
- 互換 CLI フロント

### 9.3 MVP 非対象

- streaming
- Web UI
- server mode 本実装
- 会議 SaaS 連携
- 高度な整形機能
- NeMo backend

---

## 10. 互換性要件

### 10.1 基準バージョン

WhisperX 互換性の基準は **WhisperX v3.8.1** とする。

### 10.2 互換方針

保証するのは **UX 互換** であり、内部実装互換ではない。

### 10.3 互換対象

以下を MVP の互換対象とする。

- CLI の主要オプション群
- 出力形式: `json`, `srt`, `vtt`, `txt`
- JSON 出力の主要フィールド
- WhisperX 利用者の移行に必要な主要挙動

### 10.4 非互換対象

- WhisperX の内部モジュール構造
- WhisperX 固有の依存解決挙動
- WhisperX と同一のインポートパス互換

### 10.5 差分台帳

Appendix B には、少なくとも以下を含む差分台帳を記載する。

- WhisperX v3.8.1 側の仕様
- whisper-Ω 側の仕様
- 同等 / 部分互換 / 非互換 の判定
- 差分理由
- 移行時の注意事項

### 10.6 受け入れ基準

- 互換対象 CLI オプションについて、Appendix B で定義したテストケースの 100% を通過すること
- 互換対象 JSON フィールドについて、Appendix A/B に定義した契約を満たすこと
- 互換対象の shell-level 終了結果について、Appendix B/C で定義した exit status family の対応表に従うこと

---

## 11. diarization 要件

### 11.1 方針

diarization は plugin / backend として分離し、本体必須依存にしない。

### 11.2 MVP backend

- `none`
- `pyannote`

### 11.3 出力契約

以下の意味論を保証する。

- `segments[].speaker`
- `words[].speaker`
- `speakers`
- 実行に使用した `diarization_backend`
- diarization 実行可否および degraded 情報

### 11.4 未実行時の表現

diarization 未実行、未利用、または話者不明時は、以下のルールを適用する。

- `segments[].speaker` は **必ず存在**し、値は `null` とする
- `words[].speaker` は **words 要素が存在する場合は必ず存在**し、値は `null` とする
- 空文字は使用しない
- `speakers` は空配列 `[]` とする
- フィールド省略は行わない

このルールは Appendix A の JSON Schema と Appendix B の互換試験で拘束力を持つ。

---

## 12. アーキテクチャ要件

### 12.1 構成方針

monorepo で開始し、内部責務を以下の単位に分離する。

- `asr`
- `vad`
- `align`
- `diarize`
- `merge`
- `io`
- `runtime`
- `compat`
- `cli`

### 12.2 backend 抽象化

少なくとも以下を抽象化対象とする。

- `ASRBackend`
- `AlignmentBackend`
- `DiarizationBackend`
- `VADBackend`

詳細な interface は Appendix F で定義する。

---

## 13. 実行ポリシーと失敗モデル

### 13.1 結果分類

結果は以下に分類する。

- `success`
- `degraded`
- `failure`

`degraded` は、要求された一部機能または要求品質を満たさずに処理が継続した状態を示す。

### 13.2 デフォルト挙動

本文におけるデフォルトは以下とする。

- CLI 単発実行のデフォルト `runtime_policy` は `permissive`
- 非対話バッチ、API、将来の server mode のデフォルト `runtime_policy` は `strict`
- `--device cuda` 指定時は、`permissive` であっても CPU への自動降格を禁止する
- `--device auto` 指定時に限り、`permissive` では CPU への降格を許容できる
- `strict-gpu` は GPU 利用を必須とし、GPU 要求未達時は失敗とする

本書では `production default` という曖昧な語を用いず、上記のデフォルト定義を正式仕様とする。

### 13.3 degraded / failure 時の機械可読性

`status=degraded` または `status=failure` の場合、可能な限り以下を返すこと。

- `schema_version`
- `status`
- `metadata`
- `error_code`
- `error_category`
- `backend_errors[]`

`metadata` には最低限以下を含める。

- `requested_features`
- `completed_features`
- `failed_features`
- `requested_device`
- `actual_device`
- `fallbacks`

### 13.4 error_code / error_category の規則

以下を正式規則とする。

- `success` の場合、`error_code=null`、`error_category=null`、`backend_errors=[]`
- `degraded` の場合、`error_code` および `error_category` は **null / non-null の両方を許容**する
- `degraded` で backend 起因の降格、部分失敗、代替実行が発生した場合、`backend_errors[]` には原因要素を記録する
- `degraded` で単一代表原因を付与可能な場合、`error_code` と `error_category` を non-null としてよい
- `failure` の場合、`error_code` と `error_category` は原則必須とする
- `failure` の場合、`backend_errors[]` は可能な限り保持する

`error_code` は「失敗専用コード」ではなく、「劣化・失敗を含む異常原因コード」として扱う。  
`error_code` は `omega doctor` の既知原因コード体系と整合することが望ましい。正式なコード体系は Appendix D/H で固定する。

### 13.5 exit code

CLI の exit code は以下を正式値とする。

- `0`: 完全成功
- `10`: degraded success
- `20`: runtime failure
- `30`: dependency failure
- `31`: configuration failure
- `40`: usage / argument error

WhisperX 側の終了結果との対応関係は Appendix B/C にマッピング表として定義する。  
MVP では数値一致ではなく shell-level の成功/失敗意味論の互換を重視する。

### 13.6 runtime policy

以下の policy を持つ。

- `permissive`
- `strict`
- `strict-gpu`

意味は以下の通り。

- `permissive`: 品質低下または機能欠落を許容して継続可能
- `strict`: 要求未達時は失敗とする
- `strict-gpu`: `strict` に加え、GPU 要求未達時に CPU へ降格しない

### 13.7 CLI 入力契約

CLI では以下の正式入力を持つ。

- `--runtime-policy {permissive,strict,strict-gpu}`
- `--require-feature <feature>` （複数指定可）
- `--require-diarization`
- `--require-alignment`
- `--device {auto,cpu,cuda}`
- `--emit-result-json {auto,always,on-failure,never}`
- `--write-failure-json`

`--require-diarization` は `--require-feature diarization` の糖衣構文とする。  
`--require-alignment` は `--require-feature alignment` の糖衣構文とする。  
`--runtime-policy strict-gpu` と `--device cpu` の組み合わせは usage error とする。

### 13.8 Python API 入力契約

Python API では、少なくとも以下と意味論互換な入力を提供すること。

- `runtime_policy`
- `required_features`
- `device`

詳細な API 契約、例外方針、主要ユースケース単位の期待挙動は Appendix E で定義する。

### 13.9 CLI failure 時の出力契約

CLI は以下の出力契約を持つ。

- 人間向けログ、警告、診断メッセージは `stderr` に出力する
- 機械可読な結果 JSON は `stdout` に出力可能であること
- `--emit-result-json=always` 指定時は、`success` / `degraded` / `failure` のいずれでも JSON を `stdout` に出力する
- `--emit-result-json=on-failure` 指定時は、`failure` 時に JSON を `stdout` に出力する
- `--emit-result-json=never` 指定時は、機械可読 JSON を `stdout` に出力しない
- `--output-file` 等でファイル出力先が指定されている場合、`success` / `degraded` の結果は指定先へ書き出す
- `failure` 時にファイルへ機械可読 JSON を書き出すのは、`--write-failure-json` または Appendix C で定義する等価指定がある場合に限る
- `failure` 時、JSON を出力しないモードであっても `stderr` には簡潔な失敗理由を出すこと

デフォルトの `--emit-result-json` は `auto` とし、MVP では CLI 単体利用の可読性を優先して Appendix C で具体挙動を固定する。

### 13.10 doctor と self-check

MVP で必須とするのは `omega doctor` によるローカル実行環境診断である。  
本書でいう health check は server mode を前提とした将来用語とし、MVP の必須実装には含めない。  
MVP では `doctor` および必要に応じて `self-check` を用語として用いる。

---

## 14. インストール・配布要件

### 14.1 基本方針

`pip install` 一発で全環境・全機能を完全自動化することは目標にしない。  
代わりに、失敗しにくい導線を正式提供する。

### 14.2 導線

- `pip install whisper-omega`
- `uv tool install whisper-omega`
- extras による optional 機能導入
- Docker image

### 14.3 必須導線

以下を README 先頭に記載可能な状態にすること。

- 転写だけ行う最短手順
- `omega doctor`
- `omega setup core`
- Docker での最短手順

---

## 15. 性能要件

### 15.1 基本方針

性能評価は、絶対値だけでなく固定ベースライン比較でも判定する。

### 15.2 ベースライン

MVP では性能比較ベースラインを以下に固定する。

- WhisperX: v3.8.1
- faster-whisper: v1.2.1
- CTranslate2: Appendix D に固定記載
- 評価モデル: Appendix D に固定記載
- 評価データセット: Appendix D に固定記載

「現行安定版」等の動的表現は使用しない。

### 15.3 受け入れ基準

第一級サポート対象環境において、Appendix D の固定条件で以下を満たすこと。

- 同一モデル・同一デバイス条件で、WhisperX v3.8.1 比の処理時間が 1.20 倍以内
- 同一条件で、faster-whisper v1.2.1 直利用比の処理時間が 1.35 倍以内
- 60 分音声のバッチ処理で、非異常終了率 99%以上
- `strict-gpu` 時に GPU 要求未達なら silent CPU fallback を行わないこと
- OOM 発生時、usage error ではなく runtime failure として分類されること

### 15.4 測定手順

測定は Appendix D に従う。最低限、以下を定義すること。

- 測定データセット
- 試行回数
- 採用統計値
- ばらつき記録方法
- 再実行条件
- 測定失敗時の扱い
- 測定環境固定条件
- warm cache / cold cache の区別
- モデル download 時間を含めるかの定義
- 初回モデルロード時間を別計測にするかの定義
- 出力書き込み時間を含めるかの定義
- VAD on/off の扱い
- ASR only / alignment / diarization を別ケースとして扱う定義

MVP の既定値として、性能試験は各ケース 5 回実施し、代表値は median を採用する。  
mean は参考値として記録してもよいが、受け入れ判定には使用しない。

### 15.5 受け入れ判定に用いる benchmark 種別

Appendix D では benchmark を少なくとも以下に分けること。

- **Setup benchmark**: download / setup を含む
- **Cold-start benchmark**: model download を除外し、初回ロードを含む
- **Steady-state benchmark**: model preload 済み、download / 初回初期化を除外

MVP の性能受け入れ判定は、原則として **Steady-state benchmark** を基準とする。  
Cold-start benchmark は参考値として管理する。

---

## 16. 出力契約

### 16.1 方針

外部契約としての JSON 出力を安定化する。  
詳細 schema は Appendix A で定義する。

### 16.2 schema version

JSON 出力には必ず `schema_version` を含める。  
MVP 正式版では `1.0.0` を使用する。

### 16.3 必須トップレベルフィールド

以下を必須とする。

- `schema_version`
- `status`
- `text`
- `language`
- `segments`
- `words`
- `speakers`
- `metadata`

`status=failure` の場合でも、可能な限り上記を返す。

### 16.4 metadata の必須フィールド

以下を必須とする。

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

### 16.5 null / 空配列方針

- `speakers` が存在しない状態は認めない
- diarization 未実行時の `speakers` は `[]`
- `segments` / `words` は空配列を許容する
- `speaker` は `null` を許容する

### 16.6 failure / degraded 時の追加フィールド

`status=degraded` または `status=failure` 時は、可能な限り以下を含める。

- `error_code`
- `error_category`
- `backend_errors`

### 16.7 丸め規則

時刻・信頼度・数値表現の丸め規則は Appendix A/G で固定する。

---

## 17. Python API 要件

### 17.1 方針

Python API については WhisperX との**意味論互換**を目指す。構造互換は保証しない。

### 17.2 意味論互換の定義

本書における意味論互換とは、少なくとも以下が一致または整合することを指す。

- 同一ユースケースで同種の入力を受け取れること
- 同一ユースケースで同等意味の結果を返せること
- 要求未達時の失敗種別が文書化されていること
- strict/permissive の差異が一貫していること

### 17.3 別紙化要件

Appendix E では少なくとも以下を定義すること。

- API entrypoint
- 入力契約
- 出力契約
- 例外方針
- 主要ユースケース別期待挙動

---

## 18. 可観測性・診断要件

### 18.1 MVP 必須

- `omega doctor`
- degraded / failure の machine-readable 表現
- backend / device / fallback 情報の出力
- known failure category の識別

### 18.2 KPI

第一級サポート環境において、以下を満たすこと。

- documented path でのセットアップ成功率 95%以上
- `omega doctor` により、既知セットアップ失敗の 90%以上を既知原因コードへ分類できること
- `strict` / `strict-gpu` で、要求未達時に silent degrade しない率 100%
- degraded 実行時に `status=degraded` または等価情報が欠落しない率 100%

### 18.3 検証

測定手順は Appendix D に定義する。

---

## 19. 受け入れ判定

MVP の受け入れ条件は以下とする。

- 本書の MUST 要件を満たすこと
- P0 別紙が承認済みであること
- 第一級サポート環境において KPI を満たすこと
- 互換試験、性能試験、診断試験を Appendix D に従って通過すること

---

## 20. 今後の作業順序

1. Appendix A: JSON Schema 本体
2. Appendix B: 互換性マトリクス
3. Appendix C: CLI 詳細仕様
4. Appendix D: Validation Protocol
5. Appendix E: Python API 契約
6. Appendix F: backend interface 定義
7. Appendix G 以降の補助仕様

---

## 21. 一文要約

`whisper-Ω` は、faster-whisper を中核に、alignment と diarization を交換可能 backend として統合し、WhisperX 互換 UX を提供しつつ、導入容易性・依存分離・可観測性・運用制御で優位性を持つ実務向け転写基盤である。
