# Validation Checklist

最終更新: 2026-03-20

この文書はローカルで回せる確認項目をまとめたもの。正式な受け入れ試験は `whisper-omega-plan/specs/appendix_d_validation_v01.md` を親とする。

## 実施済み

- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `PYTHONPATH=src python3 -m whisper_omega doctor --json-output`
- `PYTHONPATH=src python3 -m whisper_omega setup core`
- `PYTHONPATH=src python3 -m whisper_omega transcribe <tmp.wav> --device cpu --emit-result-json always`
- `.venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --output-format json --emit-result-json always`
- `MPLCONFIGDIR=/tmp/mpl .venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --require-diarization --diarize-backend pyannote --output-format json --emit-result-json always`

## 現在の期待結果

- テストは green
- `doctor` は Python / platform / `faster-whisper` / `nvidia-smi` の情報を返す
- `setup core` は documented path を返す
- `transcribe` は `faster-whisper` 未導入環境では `DEPENDENCY_MISSING` の JSON failure を返す
- `faster-whisper` 導入後は CPU 上で end-to-end smoke test が成功する
- `pyannote` 導入済みで `HF_TOKEN` 未設定時は `HF_TOKEN_MISSING` の degraded JSON を返す

## 次に追加したい検証

- `faster-whisper` 導入後の実音声 transcribe smoke test
- `--output-file` に対する JSON / SRT / VTT / TXT ファイル生成確認
- `whisperx` 互換フロントのオプションマッピング確認
- strict / permissive / strict-gpu の device matrix 確認
- Appendix B の CLI / JSON / EXIT ケースの自動実行
