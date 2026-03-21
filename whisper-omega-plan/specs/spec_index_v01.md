# whisper-Ω 仕様書パッケージ v0.1

## 内容
- `appendix_a_schema_v01.md`
- `appendix_a_schema_v01.json`
- `appendix_b_compat_v01.md`
- `appendix_c_cli_v01.md`
- `appendix_d_validation_v01.md`

## 位置づけ
親要件文書 v0.5 を補完する実装拘束用の仕様群。P0 優先の別紙を中心にまとめた。

## 優先度
1. Appendix A
2. Appendix C
3. Appendix B
4. Appendix D

## 確認したい点
- language code を BCP-47 に厳密化するか
- `--diarize` を正式 alias として持つか
- failure 時 JSON をデフォルトでファイル出力させるか
- validation データセットの実ファイル候補
