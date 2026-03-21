# Dataset Manifest Template

最終更新: 2026-03-21

使い方:
- `python3 scripts/build_dataset_manifest.py <dataset-dir> --dataset-id D1_SHORT_JA`
- 出力をこの表の該当 dataset 行へ貼り付ける
- `Language`, `Speakers`, `Notes` は人手で補完する
- 候補データセットは `docs/VALIDATION_DATASET_CANDIDATES.md` を参照
- 現在の D1/D2 固定値は `docs/VALIDATION_DATASET_MANIFEST.md` を参照

| Dataset ID | File | SHA256 | Duration | Language | Speakers | Notes |
|---|---|---|---|---|---|---|
| D1_SHORT_JA | | | | ja | 1 | |
| D2_SHORT_EN | | | | en | 1 | |
| D3_LONG_MIXED | | | | mixed | | |
| D4_DIARIZATION | | | | mixed | 2-4 | |
| D5_FAILURE_INJECTION | | | | n/a | n/a | |
