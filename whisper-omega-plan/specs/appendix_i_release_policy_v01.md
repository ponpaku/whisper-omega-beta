# Appendix I: リリース / 互換性維持ポリシー v0.1

## 1. 目的

MVP 以降の互換性維持と破壊的変更の扱いを明文化する。

## 2. 契約面の基本原則

- JSON 契約は `schema_version` で管理する
- 互換破壊はメジャー変更でのみ許可する
- CLI の既存主要オプションは少なくとも 1 minor は互換 alias を維持する

## 3. backend 面の原則

- optional backend の追加は minor で許可する
- backend 廃止は deprecation notice を経て行う

## 4. テスト面の原則

- Appendix B の互換試験は継続実行する
- Appendix D の validation 手順はリリース候補で再実行する

## 5. 文書更新規則

- 要件変更時は親文書と対応別紙を同時更新する
- open decisions は `DECISIONS.md` から正式仕様へ昇格させる

