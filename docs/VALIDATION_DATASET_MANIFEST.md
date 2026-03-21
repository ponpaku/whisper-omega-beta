# Validation Dataset Manifest

Generated on 2026-03-21 from `google/fleurs` validation split using `scripts/export_google_fleurs_fixtures.py` and `scripts/build_dataset_manifest.py`.

Local fixture directories:
- `fixtures/d1_short_ja`
- `fixtures/d2_short_en`
- `fixtures/d3_long_mixed`
- `fixtures/d4_diarization`
- `fixtures/d5_failure_injection`

## D1_SHORT_JA

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D1_SHORT_JA | 10411584430488337925.wav | 091c2c7fc285593a60dc4b90db1c1c20cffa5a0551c8eb2e26df687dffbb1fd8 | 12.120 | ja | 1 | Exported from `google/fleurs` `ja_jp` `dev.tsv` |
| D1_SHORT_JA | 11946010384058816161.wav | 6c31230ceb5cbabde32f9ed4074ad26235eafba13de7f7ca9c0f7b1721015e94 | 16.740 | ja | 1 | Exported from `google/fleurs` `ja_jp` `dev.tsv` |
| D1_SHORT_JA | 399340006455025662.wav | ba7d6a36c5dcc419b193ae441be35139e53c703356861e0a6262776567b93daf | 10.800 | ja | 1 | Exported from `google/fleurs` `ja_jp` `dev.tsv` |
| D1_SHORT_JA | 8589356000221302602.wav | 69726dd0820dbe415cc42ca33244b3a398191850d3d136d63c6de04a587c3188 | 10.680 | ja | 1 | Exported from `google/fleurs` `ja_jp` `dev.tsv` |
| D1_SHORT_JA | 8942640175181466228.wav | 972308e93d8985f9ff4cdcbc6518c0057176f5f4f172829502ecd160f9f6f953 | 18.540 | ja | 1 | Exported from `google/fleurs` `ja_jp` `dev.tsv` |

## D2_SHORT_EN

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D2_SHORT_EN | 12952903060751652532.wav | 8894173d97fed19f3a7eba7ae5e9ee482270a527b961c6f4ef769215bb2ac46c | 10.580 | en | 1 | Exported from `google/fleurs` `en_us` `dev.tsv` |
| D2_SHORT_EN | 15158676295442294624.wav | d9e7df4e15666b1f5ff61b0221c714a97e815392fbf3cbc7a79c737564f801bd | 11.460 | en | 1 | Exported from `google/fleurs` `en_us` `dev.tsv` |
| D2_SHORT_EN | 16131823300806444840.wav | 5edc6f643502d49f46438a222841e621071eb2f50a6016a1c21e3d1dcc5b0fb6 | 12.840 | en | 1 | Exported from `google/fleurs` `en_us` `dev.tsv` |
| D2_SHORT_EN | 2606692427476446963.wav | 1c933543b1c3933273705557f353752f040da486eed6bc90d854ceb32c9b92d6 | 9.120 | en | 1 | Exported from `google/fleurs` `en_us` `dev.tsv` |
| D2_SHORT_EN | 2812938565630042744.wav | 409d5cfd020e15c4cbe9268c46bb7877d6ce2d5ed1470312bddc3695c5587529 | 9.120 | en | 1 | Exported from `google/fleurs` `en_us` `dev.tsv` |

## D3_LONG_MIXED

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D3_LONG_MIXED | d3_concat_01.wav | cf16fac9223034c2430f1b58bfd0cf706d0cd61163297308e305d5f8d660cd63 | 52.400 | ja+en | 1 | Concatenated from D1/D2 fixture wavs with `gap_ms=500`; see `fixtures/d3_long_mixed/d3_concat_01.recipe.json` |
| D3_LONG_MIXED | d3_concat_02.wav | 1cf72f3f8e8aa082e21ef72b780e75094a7181037eda968765e09c52e10088ba | 42.000 | ja+en | 1 | Concatenated from alternate D1/D2 fixture wavs with `gap_ms=750`; see `fixtures/d3_long_mixed/d3_concat_02.recipe.json` |

## D4_DIARIZATION

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D4_DIARIZATION | d4_mix_01.wav | f0e0d96a425acf924e457b71a3b494f20b2ffe5b33a7f8f7d7b58e2333028ff7 | 12.660 | mixed | 2 | Mixed from `12952903060751652532.wav` and `15158676295442294624.wav`; track offsets `0,1200`; see `fixtures/d4_diarization/d4_mix_01.recipe.json` |
| D4_DIARIZATION | d4_mix_overlap_01.wav | ebb12e4c11444434ff13c05bf583861ee70c7d0cd0119ef0c72d48904bc556a7 | 11.760 | mixed | 2 | Strong-overlap mix from `12952903060751652532.wav` and `15158676295442294624.wav`; track offsets `0,300`; see `fixtures/d4_diarization/d4_mix_overlap_01.recipe.json` |
| D4_DIARIZATION | d4_mix_3spk_01.wav | d48e87aa7bfd28d070cbb19a4f2fd6db946d53505c8349564c10495a207f470e | 14.640 | mixed | 3 | Three-speaker mix from `12952903060751652532.wav`, `15158676295442294624.wav`, and `16131823300806444840.wav`; track offsets `0,900,1800`; see `fixtures/d4_diarization/d4_mix_3spk_01.recipe.json` |

## D5_FAILURE_INJECTION

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D5_FAILURE_INJECTION | empty.wav | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |  | n/a | n/a | Expected `AUDIO_DECODE_FAILURE`; zero-byte wav |
| D5_FAILURE_INJECTION | truncated.wav | 00d7638a8265ede65d8caba8969f95933d69110393bed30a46f76dfce5d153a0 |  | n/a | n/a | Expected `AUDIO_DECODE_FAILURE`; truncated from `10411584430488337925.wav` |
| D5_FAILURE_INJECTION | not_audio.wav | 4308ff42d168ed56f702ee44c7b1c7f18d5de853ba1837c3012b4ca4bfeb32dd |  | n/a | n/a | Expected `AUDIO_DECODE_FAILURE`; plain text with wav extension |
| D5_FAILURE_INJECTION | missing.wav | n/a |  | n/a | n/a | Expected `AUDIO_DECODE_FAILURE`; intentionally absent path |
| D5_FAILURE_INJECTION | permission_denied_output_dir | scenario |  | n/a | n/a | Expected `OUTPUT_PERMISSION_DENIED`; use a non-writable output directory while writing result files |
| D5_FAILURE_INJECTION | dependency_missing_core | scenario |  | n/a | n/a | Expected `DEPENDENCY_MISSING`; run without `faster-whisper` installed |
