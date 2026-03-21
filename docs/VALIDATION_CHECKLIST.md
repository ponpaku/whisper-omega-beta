# Validation Checklist

最終更新: 2026-03-21

この文書はローカルで回せる確認項目をまとめたもの。正式な受け入れ試験は `whisper-omega-plan/specs/appendix_d_validation_v01.md` を親とする。

## 実施済み

- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `python3 -m pip install '.[validation]'`
- `PYTHONPATH=src python3 -m whisper_omega doctor --json-output`
- `python3 scripts/build_dataset_manifest.py <dataset-dir> --dataset-id D1_SHORT_JA`
- `python3 scripts/export_google_fleurs_fixtures.py D1_SHORT_JA fixtures/d1_short_ja --count 5`
- `python3 scripts/export_google_fleurs_fixtures.py D2_SHORT_EN fixtures/d2_short_en --count 5`
- `python3 scripts/build_long_fixture.py fixtures/d3_long_mixed/d3_concat_01.wav fixtures/d1_short_ja/10411584430488337925.wav fixtures/d2_short_en/12952903060751652532.wav fixtures/d1_short_ja/11946010384058816161.wav fixtures/d2_short_en/15158676295442294624.wav --gap-ms 500`
- `python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/12952903060751652532.wav fixtures/d2_short_en/15158676295442294624.wav fixtures/d4_diarization/d4_mix_01.wav --offset-ms 1200`
- `python3 scripts/build_failure_fixtures.py fixtures/d1_short_ja/10411584430488337925.wav fixtures/d5_failure_injection`
- `python3 scripts/generate_validation_report.py --output validation-report.json`
- `PYTHONPATH=src python3 -m whisper_omega setup core`
- `PYTHONPATH=src python3 -m whisper_omega setup validation`
- `PYTHONPATH=src python3 -m whisper_omega transcribe <tmp.wav> --device cpu --emit-result-json always`
- `PYTHONPATH=src python3 -m whisper_omega transcribe <tmp.wav> --device cpu --require-alignment --align-backend wav2vec2 --emit-result-json always`
- `.venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --output-format json --emit-result-json always`
- `MPLCONFIGDIR=/tmp/mpl .venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --require-diarization --diarize-backend pyannote --output-format json --emit-result-json always`

## 現在の期待結果

- テストは green
- `doctor` は Python / platform / `faster-whisper` / `ctranslate2` / `torch` / `torchaudio` / `torchcodec` / `nvidia-smi` / alignment・diarization readiness / decode backend / recommended actions を返す
- `build_dataset_manifest.py` は dataset file / SHA256 / duration の markdown 行を生成できる
- `export_google_fleurs_fixtures.py` は `google/fleurs` から D1/D2 候補 fixture をローカル wav に書き出せる
- `build_long_fixture.py` は既存 fixture から D3_LONG_MIXED 向け long-form wav と recipe JSON を作れる
- `build_diarization_fixture.py` は D4_DIARIZATION 向けの synthetic mixture と recipe JSON を作れる
- `build_failure_fixtures.py` は D5_FAILURE_INJECTION 向けの decode failure fixture を作れる
- `generate_validation_report.py` は doctor / unittest / smoke の結果を JSON へまとめられる
- `docs/VALIDATION_DATASET_CANDIDATES.md` は Google-first の dataset 候補を示す
- `docs/VALIDATION_DATASET_MANIFEST.md` は D1/D2 の実際の local fixture hash / duration を固定する
- `setup core` は documented path を返す
- `setup validation` は validation 固定化の手順を返す
- `transcribe` は `faster-whisper` 未導入環境では `DEPENDENCY_MISSING` の JSON failure を返す
- `faster-whisper` 導入後は CPU 上で end-to-end smoke test が成功する
- `align` extra 導入後は latin-script 音声に対して `wav2vec2` alignment が成功する
- `OMEGA_ALIGNMENT_ROMANIZER` 設定時は非 latin transcript を romanize して alignment に流せる
- `pyannote` 導入済みで `HF_TOKEN` 未設定時は `HF_TOKEN_MISSING` の degraded JSON を返す
- diarization では decode stack として `torchaudio` または `ffmpeg`+`torchcodec` の readiness が確認でき、`torchaudio` 利用時は in-memory waveform 経路も使える
- D1/D2 の実データ fixture は `fixtures/d1_short_ja` / `fixtures/d2_short_en` にローカル export 済み
- D3 のローカル fixture は `fixtures/d3_long_mixed` に生成できる
- D4/D5 のローカル fixture は `fixtures/d4_diarization` / `fixtures/d5_failure_injection` に生成できる

## 次に追加したい検証

- `faster-whisper` 導入後の実音声 transcribe smoke test
- `align` extra 導入後の英語短尺音声 forced alignment smoke test
- `--output-file` に対する JSON / SRT / VTT / TXT ファイル生成確認
- `whisperx` 互換フロントのオプションマッピング確認
- strict / permissive / strict-gpu の device matrix 確認
- Appendix B の CLI / JSON / EXIT ケースの自動実行
