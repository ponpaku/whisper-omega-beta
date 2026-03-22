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
- `python3 scripts/build_long_fixture.py fixtures/d3_long_mixed/d3_concat_02.wav fixtures/d2_short_en/16131823300806444840.wav fixtures/d1_short_ja/8942640175181466228.wav fixtures/d2_short_en/2812938565630042744.wav --gap-ms 750`
- `python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/12952903060751652532.wav fixtures/d2_short_en/15158676295442294624.wav fixtures/d4_diarization/d4_mix_01.wav --offset-ms 1200`
- `python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/12952903060751652532.wav fixtures/d2_short_en/15158676295442294624.wav fixtures/d4_diarization/d4_mix_overlap_01.wav --offset-ms 300`
- `python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/12952903060751652532.wav fixtures/d2_short_en/15158676295442294624.wav fixtures/d4_diarization/d4_mix_3spk_01.wav --offset-ms 900 --speaker-c fixtures/d2_short_en/16131823300806444840.wav --offset-ms-c 1800`
- `python3 scripts/build_failure_fixtures.py fixtures/d1_short_ja/10411584430488337925.wav fixtures/d5_failure_injection`
- `python3 scripts/benchmark_smoke.py tmp_smoke.wav --model tiny --repeats 2 --devices cpu cuda`
- `python3 scripts/generate_validation_report.py --output validation-report.json`
- `python3 scripts/generate_validation_report.py --include-smoke --output validation-report.json`
- `python3 scripts/generate_validation_report.py --include-alignment-smoke --output validation-report.json`
- `python3 scripts/generate_validation_report.py --include-alignment-smoke --include-diarization-smoke --output validation-report.json`
- `python3 scripts/generate_validation_report.py --include-smoke --include-alignment-smoke --include-diarization-smoke --output validation-report.json`
- `PYTHONPATH=src python3 -m whisper_omega setup core`
- `PYTHONPATH=src python3 -m whisper_omega setup validation`
- `PYTHONPATH=src python3 -m whisper_omega transcribe <tmp.wav> --device cpu --emit-result-json always`
- `PYTHONPATH=src python3 -m whisper_omega transcribe <tmp.wav> --device cpu --require-alignment --align-backend wav2vec2 --emit-result-json always`
- `.venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --output-format json --emit-result-json always`
- `.venv-system/bin/omega transcribe stereo_sample.wav --device cpu --model tiny --require-diarization --diarize-backend channel --output-format json --emit-result-json always`
- `.venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --require-diarization --diarize-backend nemo --output-format json --emit-result-json always`
- `MPLCONFIGDIR=/tmp/mpl .venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --require-diarization --diarize-backend pyannote --output-format json --emit-result-json always`
- `OMEGA_PYANNOTE_MIN_SPEAKERS=2 OMEGA_PYANNOTE_MAX_SPEAKERS=3 MPLCONFIGDIR=/tmp/mpl .venv-system/bin/omega transcribe tmp_smoke.wav --device cpu --model tiny --require-diarization --diarize-backend pyannote --output-format json --emit-result-json always`

## 現在の期待結果

- テストは green
- `doctor` は Python / platform / `faster-whisper` / `ctranslate2` / `torch` / `torchaudio` / `torchcodec` / `nvidia-smi` / alignment・diarization readiness / decode backend / alignment map 設定 / pyannote speaker hint / recommended actions を返す
- `build_dataset_manifest.py` は dataset file / SHA256 / duration の markdown 行を生成できる
- `export_google_fleurs_fixtures.py` は `google/fleurs` から D1/D2 候補 fixture をローカル wav に書き出せる
- `build_long_fixture.py` は既存 fixture から D3_LONG_MIXED 向け long-form wav と recipe JSON を作れる
- `build_diarization_fixture.py` は D4_DIARIZATION 向けの synthetic mixture と recipe JSON を作れる
- `build_failure_fixtures.py` は D5_FAILURE_INJECTION 向けの decode failure fixture を作れる
- `generate_validation_report.py` は doctor / unittest / ASR smoke / alignment smoke / diarization smoke の結果を JSON へまとめられる
- `run_alignment_smoke.py` は `fixtures/d2_short_en/manifest.json` と `fixtures/d1_short_ja/manifest.json` を使って alignment routing smoke を返せる
- `run_diarization_smoke.py` は `fixtures/d4_diarization/*.recipe.json` を使って `pyannote` / `nemo` の diarization assignment smoke を返せる
- `docs/VALIDATION_DATASET_CANDIDATES.md` は Google-first の dataset 候補を示す
- `docs/VALIDATION_DATASET_MANIFEST.md` は D1/D2 の実際の local fixture hash / duration を固定する
- `setup core` は documented path を返す
- `setup validation` は validation 固定化の手順を返す
- `transcribe` は `faster-whisper` 未導入環境では `DEPENDENCY_MISSING` の JSON failure を返す
- `faster-whisper` 導入後は CPU 上で end-to-end smoke test が成功する
- `align` extra 導入後は latin-script 音声と kana-only Japanese に対して `wav2vec2` alignment が成功する
- alignment の token resolution は `OMEGA_ALIGNMENT_TEXT_MAP` -> latin native / Japanese kana or reading map -> `OMEGA_ALIGNMENT_ROMANIZER` -> unsupported の順で固定されている
- `OMEGA_ALIGNMENT_JA_READING_MAP` 設定時は kanji を含む日本語 transcript を読み仮名経由で alignment に流せる
- `OMEGA_ALIGNMENT_TEXT_MAP` 設定時は任意言語の語を normalized token へ上書きできる
- `scripts/build_ja_reading_map.py <manifest.json> --output ja_reading_map.json` で読み辞書のたたき台を作れる
- `scripts/build_alignment_text_map.py <manifest.json> --output text_map.json` で多言語用 text map のたたき台を作れる
- `OMEGA_ALIGNMENT_ROMANIZER` 設定時はその他の非 latin transcript を romanize して alignment に流せる
- unsupported transcript は `ALIGNMENT_LANGUAGE_UNSUPPORTED` / `ALIGNMENT_TEXT_UNSUPPORTED` / map validation code として機械可読に返る
- `pyannote` 導入済みで `HF_TOKEN` 未設定時は `HF_TOKEN_MISSING` の degraded JSON を返す
- diarization では decode stack として `torchaudio` または `ffmpeg`+`torchcodec` の readiness が確認でき、`torchaudio` 利用時は in-memory waveform 経路も使える
- built-in `channel` diarization backend は stereo wav を `CHANNEL_LEFT` / `CHANNEL_RIGHT` へ割り当てられる
- `nemo` diarization backend は manifest + RTTM 経路で speaker assignment を返せる
- diarization では `OMEGA_PYANNOTE_NUM_SPEAKERS` / `MIN_SPEAKERS` / `MAX_SPEAKERS` を speaker hint として backend へ渡せる
- diarization failure は `HF_TOKEN_MISSING` / `DIARIZATION_AUTH_FAILURE` / `DIARIZATION_MODEL_UNAVAILABLE` / `DIARIZATION_DECODE_FAILURE` / `CONFIG_INVALID` に分類される
- `nemo` diarization failure は `NEMO_MODEL_UNAVAILABLE` / `NEMO_RUNTIME_FAILURE` / `NEMO_OUTPUT_MISSING` / `CONFIG_INVALID` に分類される
- channel diarization failure は `DIARIZATION_CHANNELS_UNAVAILABLE` / `DIARIZATION_CHANNEL_AMBIGUOUS` / `DIARIZATION_AUDIO_UNSUPPORTED` に分類される
- D1/D2 の実データ fixture は `fixtures/d1_short_ja` / `fixtures/d2_short_en` にローカル export 済み
- D3 のローカル fixture は `fixtures/d3_long_mixed` に生成できる
- D4/D5 のローカル fixture は `fixtures/d4_diarization` / `fixtures/d5_failure_injection` に生成できる
- D4 は `d4_mix_01.wav` / `d4_mix_overlap_01.wav` / `d4_mix_3spk_01.wav` の 3 ケースを固定している
- D3 は `d3_concat_01.wav` / `d3_concat_02.wav` の 2 ケースを固定している
- D5 manifest には `OUTPUT_PERMISSION_DENIED` と `DEPENDENCY_MISSING` の scenario fixture も含めている

## 残留リスクとして監視する項目

- GPU 実機では decode/backend 条件により `AUDIO_DECODE_FAILURE` が残る場合がある
- non-latin alignment は map / romanizer 補助を前提とした実装であり、一般解ではない
- pyannote diarization の本番安定性は `HF_TOKEN` と依存スタックの状態に左右される
