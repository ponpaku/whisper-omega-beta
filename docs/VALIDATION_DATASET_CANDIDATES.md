# Validation Dataset Candidates

最終更新: 2026-03-21

Google 公開データセットを優先しつつ、Appendix D の D1-D5 を具体化するための候補一覧。

現時点では `D1_SHORT_JA` と `D2_SHORT_EN` の 5-file local fixture を `fixtures/` 配下へ export 済みで、hash / duration は `docs/VALIDATION_DATASET_MANIFEST.md` に固定している。

## Google-first candidates

### D1_SHORT_JA
- Source: `google/fleurs` `ja_jp`
- Purpose: 日本語 ASR / alignment 短尺検証
- Fixture rule: 10 本の 30-90 秒 fixture を、複数の短い utterance を連結して作る

### D2_SHORT_EN
- Source: `google/fleurs` `en_us`
- Purpose: 英語 ASR / alignment 短尺検証
- Fixture rule: 10 本の 30-90 秒 fixture を、複数の短い utterance を連結して作る

### D4_DIARIZATION
- Source: exported local fixture pairs from `google/fleurs`
- Purpose: diarization / speaker assignment 検証
- Fixture rule: 2-4 話者の synthetic mixture を生成する
- Note: `scripts/build_diarization_fixture.py` で local mono wav を重ねて recipe JSON を保存する

### D5_FAILURE_INJECTION
- Source candidates:
  - `speech_commands` (TensorFlow Datasets)
  - `google/fleurs`
- Purpose: decode failure / invalid argument / empty input / dependency failure の再現
- Fixture rule: 正常音声に加え、truncate / zero-byte / permission denied / missing dependency を組み合わせる

## Non-Google fallback candidate

### D3_LONG_MIXED
- Source: `facebook/voxpopuli`
- Purpose: 長尺 ASR / long-form stability / CPU-GPU steady-state 検証
- Reason: Google 公開ソースだけでは現時点で長尺連続話者音声の候補が弱いため
- Local bridge: `scripts/build_long_fixture.py` で D1/D2 fixture を連結した provisional long-form input を作れる

## Suggested acquisition commands

```bash
python3 -m pip install '.[validation]'
python3 -c "from datasets import load_dataset; load_dataset('google/fleurs', 'ja_jp', split='validation')"
python3 -c "from datasets import load_dataset; load_dataset('google/fleurs', 'en_us', split='validation')"
python3 -c "from datasets import load_dataset; load_dataset('google/xtreme_s', 'fleurs.ja_jp', split='validation')"
python3 -c "from datasets import load_dataset; load_dataset('speech_commands', split='validation')"
```

## Manifest workflow

1. Download or stream the source dataset.
2. Export the selected audio files into a local fixture directory.
3. Run `python3 scripts/build_dataset_manifest.py <fixture-dir> --dataset-id D1_SHORT_JA`.
4. Record source dataset URL and split in `Notes`.
5. Store any concatenation/mixing recipe alongside the fixture directory.
6. Copy the finalized rows into `docs/VALIDATION_DATASET_MANIFEST.md`.

## Local export helpers

```bash
python3 scripts/export_google_fleurs_fixtures.py D1_SHORT_JA fixtures/d1_short_ja --count 5
python3 scripts/export_google_fleurs_fixtures.py D2_SHORT_EN fixtures/d2_short_en --count 5
python3 scripts/build_long_fixture.py fixtures/d3_long_mixed/d3_concat_01.wav fixtures/d1_short_ja/10411584430488337925.wav fixtures/d2_short_en/12952903060751652532.wav fixtures/d1_short_ja/11946010384058816161.wav fixtures/d2_short_en/15158676295442294624.wav --gap-ms 500
python3 scripts/build_diarization_fixture.py fixtures/d2_short_en/12952903060751652532.wav fixtures/d2_short_en/15158676295442294624.wav fixtures/d4_diarization/d4_mix_01.wav --offset-ms 1200
python3 scripts/build_failure_fixtures.py fixtures/d1_short_ja/10411584430488337925.wav fixtures/d5_failure_injection
python3 scripts/build_dataset_manifest.py fixtures/d3_long_mixed --dataset-id D3_LONG_MIXED
python3 scripts/build_dataset_manifest.py fixtures/d1_short_ja --dataset-id D1_SHORT_JA
python3 scripts/build_dataset_manifest.py fixtures/d2_short_en --dataset-id D2_SHORT_EN
python3 scripts/build_dataset_manifest.py fixtures/d4_diarization --dataset-id D4_DIARIZATION
```
