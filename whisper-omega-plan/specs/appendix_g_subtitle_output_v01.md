# Appendix G: SRT / VTT 出力詳細仕様 v0.1

## 1. 目的

字幕出力のタイムスタンプ表現、行構成、MVP で保証する最小挙動を固定する。

## 2. SRT

- index は 1 始まり
- 時刻書式は `HH:MM:SS,mmm`
- 各 block は以下で構成する
  - index
  - `start --> end`
  - text

## 3. VTT

- 先頭行は `WEBVTT`
- 時刻書式は `HH:MM:SS.mmm`
- 各 cue は以下で構成する
  - 空行
  - `start --> end`
  - text

## 4. MVP のテキスト規則

- `segments[].text` をそのまま 1 cue として出力する
- 行折返しや最大文字数制御は MVP 範囲外
- speaker prefix の整形は MVP 範囲外

## 5. 丸め規則

- start / end は JSON 契約と同じく小数点以下 3 桁へ丸める
- SRT / VTT の millisecond 変換は丸めベースで行う

## 6. failure 時の扱い

- `failure` では字幕ファイルを主出力として生成しない
- `--write-failure-json` 指定時のみ JSON failure を別途書き出す

