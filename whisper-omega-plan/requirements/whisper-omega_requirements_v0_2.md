# whisper-Ω 要件定義書（ドラフト v0.2）

- 作成日: 2026-03-09
- 文書状態: Draft for Review
- 想定読者: 企画者、実装者、レビュー担当、将来のコントリビューター
- 本書の位置づけ: 構想書ではなく、MVP 受け入れ判定に利用する要件定義書

---

## 1. 文書の目的

本書は、`whisper-Ω` の MVP（最小実用版）に関する要件、受け入れ判定基準、サポート範囲、互換範囲を定義する。  
`whisper-Ω` は、WhisperX の延命版ではなく、**WhisperX の上位互換 UX を目指す新規実装**として位置付ける。

本書の目的は以下の通り。

- プロダクトの狙いを明確化する
- WhisperX の既知課題に対する改善方針を整理する
- MVP の機能範囲を定義する
- 受け入れ判定基準を明文化する
- サポート対象環境と互換保証範囲を固定する

---

## 2. 背景

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

## 3. プロダクト定義

### 3.1 位置づけ

`whisper-Ω` は以下の性質を持つ。

- WhisperX 互換 UX を持つ新規実装
- faster-whisper を中核に据えた実務向け転写基盤
- alignment / diarization を交換可能 backend として統合するオーケストレータ

### 3.2 目標

- WhisperX ユーザーが移行可能であること
- faster-whisper 直利用者に対して完成品として優位性を持つこと
- 導入時にユーザーが依存関係を推理しなくてよいこと
- 本番運用で degraded / failure / backend 状態を把握できること

### 3.3 非目標

初期段階では以下を目標にしない。

- `pip install` 一発で全機能・全環境・全 GPU 世代を完全自動化すること
- WhisperX の内部実装互換を維持すること
- 初版から全 OS / 全アーキテクチャを第一級サポートすること
- 独自 ASR モデル研究そのものを主目的にすること

---

## 4. 想定ユースケース

### 4.1 WhisperX からの移行

- 既存の WhisperX 利用者が、同等以上の機能をより安定して利用する
- WhisperX 互換 CLI / 出力形式を活かして置き換える

### 4.2 faster-whisper 直利用からの移行

- faster-whisper を直接呼び出して glue code を書いている利用者が、alignment / diarization / subtitle export をまとめて利用する

### 4.3 製品・サービスへの組み込み

- バッチ処理
- API サーバー組み込み
- 会議録生成
- 字幕生成
- 監視可能な本番運用

---

## 5. WhisperX 比の主要改善点

- 本体依存の最小化
- diarization backend の分離・交換可能化
- `doctor` / `setup` / Docker を正式導線化
- 可観測性の強化
- strict / permissive の runtime policy 導入
- degraded / failure の区別明確化
- backend 障害の局所化
- 再現可能な配布・環境管理

---

## 6. WhisperX の課題に対する整理

### 6.1 現設計で対処方針を定義済みのもの

- 依存関係の密結合
- pyannote 固定の diarization 構造
- インストール成功と実行成功のギャップ
- backend 障害が全体障害になりやすい構造
- 可観測性不足

### 6.2 対処可能だが完全には消せないもの

- CUDA / cuDNN / CTranslate2 の相性問題
- 上流依存更新の影響
- 言語別 alignment モデル事情
- 一部プラットフォームでの導入の難しさ

### 6.3 対処不可能なもの

- 外部モデル配布・認証・利用条件
- NVIDIA ランタイムの実環境依存
- 上流プロジェクトの破壊的変更そのもの
- 未整備言語向け資産が存在しない問題

### 6.4 対処不要と判断するもの

- 真の意味での完全 one-command universal install
- WhisperX の内部実装互換
- 初版からの全環境完全対応

---

## 7. サポート対象環境

### 7.1 第一級サポート対象（MVP）

MVP の正式サポート対象は以下とする。

- Python: 3.10, 3.11
- OS: Ubuntu 22.04 LTS x86_64
- CPU: x86_64 CPU 実行
- GPU: NVIDIA GPU
- CUDA 系統: CUDA 12 系
- コンテナ: Docker on Linux x86_64

### 7.2 ベストエフォート対象

以下は CI または限定検証の対象とし、正式保証は行わない。

- Python 3.12
- Windows 11 x86_64
- macOS 14+ Apple Silicon（CPU 実行のみ）
- Linux aarch64

### 7.3 非対応（MVP）

以下は MVP の対象外とする。

- CUDA 11 系での正式保証
- ROCm / AMD GPU
- Apple GPU 最適化
- 複数 GPU 分散実行
- リアルタイム配信向け低遅延保証

### 7.4 サポート境界の原則

- 第一級サポート対象のみ、受け入れ判定の母集団に含める
- ベストエフォート対象は issue 受理対象だが、SLA 相当の保証は持たない
- 非対応対象は README と `omega doctor` で明示する

---

## 8. 機能要件

### 8.1 コア機能

MVP 本体が提供すべき機能は以下。

- faster-whisper ベース ASR
- word timestamps
- JSON / SRT / VTT / TXT 出力
- CLI 提供
- Python API 提供
- model cache 管理
- CPU / GPU 両対応
- VAD の基本統合
- `omega doctor`
- `omega setup core`

### 8.2 拡張機能

以下は plugin または optional backend として扱う。

- forced alignment
- speaker diarization
- platform-native speaker attribution
- streaming
- server mode
- 高度な post-processing

### 8.3 MVP で必須の追加機能

- strict / permissive runtime policy
- `none` / `pyannote` diarization backend
- Docker CPU / CUDA image
- サンプル音声による e2e 実行

---

## 9. 互換性要件

### 9.1 互換性の基準バージョン

互換性要件の基準は **WhisperX v3.8.1** とする。

### 9.2 互換保証の対象

MVP で互換保証対象とするのは以下。

#### CLI

以下の入力・出力系オプションを互換対象とする。

- 入力音声ファイル指定
- `--model`
- `--language`
- `--output_dir`
- `--output_format`
- `--device`
- `--batch_size`
- `--vad_method`
- `--diarize`
- `--hf_token`
- `--highlight_words`

#### 出力形式

- `json`
- `srt`
- `vtt`
- `txt`

#### Python API

MVP では Python API の**構造互換ではなく意味論互換**のみを保証対象とする。

### 9.3 互換保証レベル

各項目の保証レベルは以下の 3 段階で表現する。

- **完全互換**: 同一入力・同一条件で同等意味の挙動と出力を保証
- **部分互換**: オプション名または出力意味論は互換だが、細部挙動は差異を許容
- **非互換**: 互換を目標にしない

### 9.4 MVP における互換レベル

- CLI 主要オプション: 部分互換
- 出力形式: 部分互換
- WhisperX 風 JSON 構造: 部分互換
- WhisperX 内部 API: 非互換

### 9.5 互換性テスト要件

第一級サポート環境において、互換対象 CLI オプション群の e2e テスト通過率は **100%** とする。  
ここでの通過は「同一 exit status 系統」「同一出力形式生成」「必須フィールド存在」を満たすことを意味する。

---

## 10. 話者分離要件

### 10.1 基本方針

- diarization を本体必須依存にしない
- backend 差し替え可能とする
- 出力意味論は backend に依存させない

### 10.2 想定 backend

MVP の対象は以下。

- `none`
- `pyannote`

将来対象:

- `nemo`
- `platform-native`

### 10.3 出力で保証する項目

- `segments[].speaker`
- `words[].speaker`
- `speakers`
- `metadata.diarize_backend`
- diarization の成功 / 失敗 / degraded 状態

---

## 11. runtime policy 要件

### 11.1 原則

フォールバックは無条件自動挙動ではなく、**ポリシー**として扱う。

### 11.2 モード

- `permissive`
  - 個人利用向け
  - 品質低下を許容して継続実行可能
- `strict`
  - 本番利用向け
  - 要求条件を満たせない場合は失敗
- `strict-gpu`
  - GPU 必須
  - GPU 不可時の CPU 逃がし禁止

### 11.3 原則ルール

- `--device cuda` 指定時は、production default では CPU への自動フォールバックを禁止する
- diarization 必須ジョブで diarization backend が unavailable の場合は失敗とする
- alignment 必須ジョブでは alignment backend unavailable を失敗にできる
- degraded 実行時は必ず metadata に記録する

### 11.4 結果状態

- `success`
- `degraded`
- `failure`

---

## 12. 可観測性要件

以下を本番運用前提で提供する。

- structured logs
- health check
- diagnostics
- machine-readable metadata
- backend / device / policy / fallback 状態の出力
- exit code の区別

### 12.1 必須メタデータ項目

- `requested_device`
- `actual_device`
- `requested_features`
- `completed_features`
- `failed_features`
- `fallbacks`
- `policy`
- `status`
- `asr_backend`
- `align_backend`
- `diarize_backend`

### 12.2 exit code 要件

MVP では以下を最低限満たす。

- `0`: 完全成功
- `10`: degraded 実行
- `20`: hard failure
- `30`: dependency / configuration failure

---

## 13. インストール・環境構築要件

### 13.1 基本思想

`pip install whisper-omega` のみで全機能即利用を保証することは目標にしない。  
代わりに、**失敗しにくい導線を製品として提供する**。

### 13.2 提供導線

#### 軽量導線

- `pip install whisper-omega`
- `uv tool install whisper-omega`

#### 拡張導線

- `whisper-omega[align]`
- `whisper-omega[diarize-pyannote]`
- `whisper-omega[diarize-nemo]`（将来）

#### 確実導線

- Docker image（CPU / CUDA / full）

### 13.3 必須コマンド

- `omega doctor`
- `omega setup core`
- `omega setup gpu`
- `omega setup diarize`
- `omega setup all`

### 13.4 導入原則

- ユーザーに依存関係を推理させない
- 不足要素は診断結果として提示する
- model pull / auth / device 問題を明示化する

### 13.5 導入性の受け入れ基準

第一級サポート環境において、クリーン環境から documented path に沿って `omega transcribe <sample>` まで到達する成功率は **95%以上** とする。  
測定母集団は CI 構築ジョブおよび手動検証 20 試行以上を併用する。

---

## 14. アーキテクチャ要件

### 14.1 リポジトリ方針

- 初期は monorepo とする
- 内部責務は明確に分離する

### 14.2 レイヤ構成

- `asr`
- `vad`
- `align`
- `diarize`
- `merge`
- `io`
- `runtime`
- `compat`
- `cli`

### 14.3 backend 抽象化

以下の抽象化を行う。

- `ASRBackend`
- `VADBackend`
- `AlignmentBackend`
- `DiarizationBackend`

各 backend は交換可能とし、pipeline 本体は backend 名と共通契約のみを扱う。

---

## 15. 出力契約要件

外部向け出力は内部実装より優先して安定化する。

### 15.1 versioning 方針

- すべての JSON 出力は `schema_version` を必須とする
- MVP の初期値は `1.0.0` とする
- 後方互換を壊す変更は major 更新とする

### 15.2 JSON 出力の必須フィールド

以下をトップレベル必須とする。

- `schema_version: string`
- `text: string`
- `language: string | null`
- `segments: array`
- `words: array`
- `speakers: array`
- `metadata: object`

### 15.3 `metadata` の必須フィールド

- `status: enum[success,degraded,failure]`
- `policy: string`
- `requested_device: string | null`
- `actual_device: string | null`
- `asr_backend: string`
- `align_backend: string | null`
- `diarize_backend: string | null`
- `fallbacks: array`

### 15.4 型と欠落条件の原則

- 必須フィールドは欠落させず、値が存在しない場合は `null` または空配列を用いる
- `segments` は失敗時でも空配列で存在させる
- `words` は word-level timestamp が無効または失敗した場合、空配列で存在させる
- `speakers` は diarization 未実行または失敗時、空配列で存在させる
- `status=failure` でも `metadata` は必ず返す

### 15.5 数値精度の原則

- timestamp 系数値は秒単位 float とする
- 出力時の最小精度は小数点第 3 位までを推奨する
- JSON Schema 本体では `number` として定義し、丸め規則は別紙仕様に記載する

### 15.6 別紙化する仕様

以下は本書の承認後、別紙として定義する。

- JSON Schema 本体
- SRT / VTT 出力詳細仕様
- field-level compatibility table

---

## 16. 配布要件

### 16.1 Python パッケージ

- PyPI で配布する
- optional dependencies を明示する
- `uv` lock を利用する

### 16.2 Docker

以下を提供する。

- `cpu`
- `cuda12`
- `full`

### 16.3 CI / Release

最低限以下を整備する。

- lint
- unit test
- CPU integration test
- sample audio smoke test
- wheel build
- Docker build
- release workflow

---

## 17. テスト要件

### 17.1 必須テスト

- unit tests
- integration tests
- CLI e2e tests
- golden output tests
- compat tests
- CPU smoke tests
- GPU smoke tests（第一級サポート環境のみ）

### 17.2 golden test 観点

- transcription text
- segment count
- word count
- speaker count
- subtitle segmentation
- degraded / failure の判定

### 17.3 診断品質の受け入れ基準

第一級サポート環境で意図的に注入した既知障害ケースに対して、`omega doctor` が既知原因コードを返せる率は **90%以上** とする。

---

## 18. 性能要件

### 18.1 基本方針

MVP では「絶対最速」を要求しない。  
ただし、第一級サポート環境において、WhisperX / faster-whisper 直利用に対して実務上不利にならない最低基準を設ける。

### 18.2 ベースライン

性能比較のベースラインは以下とする。

- WhisperX v3.8.1
- faster-whisper 現行安定版

比較条件は以下を揃える。

- 同一音声ファイル
- 同一モデルサイズ
- 同一 device
- 同一 batch_size
- diarization / alignment 有無を明示

### 18.3 MVP の受け入れ基準

第一級サポート環境において、以下を満たすこと。

- ASR only 条件で、同一 backend / 同一モデル時の処理時間が faster-whisper 直利用比 **+15% 以内**
- WhisperX 互換モードでの end-to-end 処理時間が WhisperX v3.8.1 比 **+20% 以内**
- 60 分音声の ASR only 実行で、OOM により異常終了しない
- 60 分音声の strict-gpu 実行で GPU 利用不可時に silent CPU fallback しない

### 18.4 リソース使用量の記録

以下を測定・保存対象とする。

- 総処理時間
- 音声長 / 処理時間比
- ピーク CPU メモリ
- ピーク GPU メモリ
- degraded 発生有無

---

## 19. MVP 定義

### 19.1 MVP で必須

- faster-whisper ベース ASR
- word timestamps
- JSON / SRT / VTT / TXT 出力
- `omega doctor`
- `omega setup core`
- WhisperX 互換の最低限 CLI
- diarization backend 抽象化
- `none` / `pyannote` backend
- strict / permissive runtime policy
- Docker CPU / CUDA image
- sample audio e2e test

### 19.2 MVP で後回し

- NeMo backend
- streaming
- server API
- Web UI
- 会議 SaaS 連携
- 高度な整形・章立て
- 全 OS / 全 GPU 世代フル対応

---

## 20. 成功条件（KPI / 受け入れ判定）

以下を満たした場合、MVP は受け入れ可能と判定する。

### 20.1 導入性

- 第一級サポート環境で documented path に沿ったセットアップ成功率 **95%以上**
- `omega doctor` による既知原因コード分類率 **90%以上**

### 20.2 安定性

- strict モードで要求未達時に silent degrade しない率 **100%**
- 互換対象 CLI e2e テスト通過率 **100%**
- 第一級サポート環境で sample audio smoke test 成功率 **100%**

### 20.3 可観測性

- `success / degraded / failure` の 3 状態がすべて machine-readable に識別可能であること
- `requested_device` と `actual_device` の両方が全ジョブ出力に存在すること
- degraded 実行時に `fallbacks` が必ず記録されること

### 20.4 互換性

- WhisperX v3.8.1 基準で、互換対象 CLI オプション群の挙動差分が文書化済みであること
- 互換対象出力形式で必須フィールド欠落率 **0%**

### 20.5 性能

- 18.3 の性能基準を満たすこと

---

## 21. 将来拡張案

- `nemo` diarization backend
- platform-native speaker attribution
- server mode / REST API / job queue
- realtime / streaming
- chaptering / summarization 連携
- SaaS 連携
- 推奨モデルセットの自動管理

---

## 22. 一文要約

`whisper-Ω` は、faster-whisper を中核に、alignment と diarization を交換可能 backend として統合し、WhisperX 互換 UX を保ちながら、導入容易性・依存分離・可観測性・運用性で勝つことを目指す実務向け転写基盤である。

---

## 23. 正式版に向けた別紙候補

- リスク一覧
- CLI 詳細仕様
- API schema
- backend interface 定義
- 監視項目一覧
- リリースポリシー
- 互換性マトリクス
- JSON Schema 本体

