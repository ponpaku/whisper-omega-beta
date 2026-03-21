# whisper-Ω 要件定義書（ドラフト v0.1）

- 作成日: 2026-03-09
- 文書状態: Draft
- 想定読者: 企画者、実装者、レビュー担当、将来のコントリビューター

---

## 1. 文書の目的

本書は、`whisper-Ω` の初期要件を定義するものである。  
`whisper-Ω` は、WhisperX の延命版ではなく、**WhisperX の上位互換を目指す新規実装**として位置付ける。

本書の目的は以下の通り。

- プロダクトの狙いを明確化する
- WhisperX の既知課題に対する改善方針を整理する
- MVP（最小実用版）の機能範囲を定義する
- 実装・運用・配布の原則を明文化する

---

## 2. 背景

WhisperX は、以下の機能を統合して提供することで実務価値を持つ。

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

- `pip install` 一発で全機能・全環境・全GPU世代を完全自動化すること
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

### 6.1 現設計で対処済み

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

## 7. 機能要件

## 7.1 コア機能

初期版で本体が提供すべき機能は以下。

- faster-whisper ベース ASR
- word timestamps
- JSON / SRT / VTT / TXT 出力
- CLI 提供
- Python API 提供
- model cache 管理
- CPU / GPU 両対応
- VAD の基本統合

## 7.2 拡張機能

以下は plugin または optional backend として扱う。

- forced alignment
- speaker diarization
- platform-native speaker attribution
- streaming
- server mode
- 高度な post-processing

## 7.3 互換性要件

### UX 互換

以下の互換を目指す。

- WhisperX 主要 CLI オプションの互換
- WhisperX 主要出力形式の互換
- WhisperX 風 JSON 構造への近似

### 非互換を許容する領域

- 内部モジュール構造
- WhisperX の依存構造
- WhisperX 固有の実装詳細

---

## 8. 話者分離要件

### 8.1 基本方針

- diarization を本体必須依存にしない
- backend 差し替え可能とする
- 出力意味論は backend に依存させない

### 8.2 想定 backend

初期想定は以下。

- `none`
- `pyannote`
- `nemo`（将来対応）
- `platform-native`（将来対応）

### 8.3 出力で保証する項目

- `segments[].speaker`
- `words[].speaker`
- `speakers`
- `metadata.diarize_backend`
- diarization の成功 / 失敗 / degraded 状態

---

## 9. runtime policy 要件

### 9.1 原則

フォールバックは無条件自動挙動ではなく、**ポリシー**として扱う。

### 9.2 モード

- `permissive`
  - 個人利用向け
  - 品質低下を許容して継続実行可能
- `strict`
  - 本番利用向け
  - 要求条件を満たせない場合は失敗
- `strict-gpu`
  - GPU 必須
  - GPU 不可時の CPU 逃がし禁止

### 9.3 原則ルール

- `--device cuda` 指定時は、production default では CPU への自動フォールバックを禁止する
- diarization 必須ジョブで diarization backend が unavailable の場合は失敗とする
- alignment 必須ジョブでは alignment backend unavailable を失敗にできる
- degraded 実行時は必ず metadata に記録する

### 9.4 結果状態

- `success`
- `degraded`
- `failure`

---

## 10. 可観測性要件

以下を本番運用前提で提供する。

- structured logs
- health check
- diagnostics
- machine-readable metadata
- backend / device / policy / fallback 状態の出力
- exit code の区別

### 10.1 期待されるメタデータ例

- requested device
- actual device
- requested features
- completed features
- failed features
- fallbacks
- policy mode
- backend 名

---

## 11. インストール・環境構築要件

## 11.1 基本思想

`pip install whisper-omega` のみで全機能即利用を保証することは目標にしない。  
代わりに、**失敗しにくい導線を製品として提供する**。

## 11.2 提供導線

### 軽量導線

- `pip install whisper-omega`
- `uv tool install whisper-omega`

### 拡張導線

- `whisper-omega[align]`
- `whisper-omega[diarize-pyannote]`
- `whisper-omega[diarize-nemo]`

### 確実導線

- Docker image（CPU / CUDA / full）

## 11.3 必須コマンド

- `omega doctor`
- `omega setup core`
- `omega setup gpu`
- `omega setup diarize`
- `omega setup all`

## 11.4 導入原則

- ユーザーに依存関係を推理させない
- 不足要素は診断結果として提示する
- できるだけ model pull / auth / device 問題を明示化する

---

## 12. アーキテクチャ要件

### 12.1 リポジトリ方針

- 初期は monorepo とする
- 内部責務は明確に分離する

### 12.2 レイヤ構成

- `asr`
- `vad`
- `align`
- `diarize`
- `merge`
- `io`
- `runtime`
- `compat`
- `cli`

### 12.3 backend 抽象化

以下の抽象化を行う。

- `ASRBackend`
- `VADBackend`
- `AlignmentBackend`
- `DiarizationBackend`

各 backend は交換可能とし、pipeline 本体は backend 名と共通契約のみを扱う。

---

## 13. 出力契約要件

外部向け出力は内部実装より優先して安定化する。

### 13.1 必須フィールド

- `text`
- `language`
- `segments`
- `words`
- `speakers`
- `metadata`

### 13.2 metadata に含めるべき情報

- `asr_backend`
- `align_backend`
- `diarize_backend`
- `requested_device`
- `actual_device`
- `policy`
- `status`
- `fallbacks`

### 13.3 設計原則

- backend を変えても意味論を崩さない
- degraded 実行を黙らせない
- 失敗を握りつぶさない

---

## 14. 配布要件

### 14.1 Python パッケージ

- PyPI で配布する
- optional dependencies を明示する
- `uv` lock を利用する

### 14.2 Docker

以下を提供する。

- `cpu`
- `cuda12`
- `full`

### 14.3 CI / Release

最低限以下を整備する。

- lint
- unit test
- CPU integration test
- sample audio smoke test
- wheel build
- Docker build
- release workflow

---

## 15. テスト要件

### 15.1 必須テスト

- unit tests
- integration tests
- CLI e2e tests
- golden output tests
- compat tests
- CPU smoke tests
- GPU smoke tests（対象環境のみ）

### 15.2 golden test 観点

- transcription text
- segment count
- word count
- speaker count
- subtitle segmentation
- degraded / failure の判定

---

## 16. MVP 定義

### 16.1 MVP で必須

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

### 16.2 MVP で後回し

- NeMo backend
- streaming
- server API
- Web UI
- 会議 SaaS 連携
- 高度な整形・章立て
- 全 OS / 全 GPU 世代フル対応

---

## 17. 成功条件

以下を満たした場合、初期段階として成功とみなす。

### 17.1 WhisperX から見た成功

- 基本機能が同等以上
- 導入が明らかに楽
- 原因不明の失敗が減る
- diarization backend 破損が全体停止に直結しない

### 17.2 faster-whisper 直利用から見た成功

- glue code を減らせる
- alignment / diarization / subtitle export をまとめて利用できる
- 本番運用向けの診断・ポリシー・可観測性がある

### 17.3 運用面の成功

- degraded と failure を区別できる
- GPU 障害を不可視化しない
- 設定と実際の実行状態を追跡できる

---

## 18. 将来拡張案

- `nemo` diarization backend
- platform-native speaker attribution
- server mode / REST API / job queue
- realtime / streaming
- chaptering / summarization 連携
- SaaS 連携
- 推奨モデルセットの自動管理

---

## 19. 一文要約

`whisper-Ω` は、faster-whisper を中核に、alignment と diarization を交換可能 backend として統合し、WhisperX 互換 UX を保ちながら、導入容易性・依存分離・可観測性・運用性で勝つことを目指す実務向け転写基盤である。

---

## 20. 参考メモ

本書は、WhisperX / faster-whisper / pyannote 系の現状調査と、本会話中で整理した設計方針をもとに作成したドラフトである。  
正式版では、以下を別紙化することを推奨する。

- リスク一覧
- CLI 詳細仕様
- API schema
- backend interface 定義
- 監視項目一覧
- リリースポリシー

