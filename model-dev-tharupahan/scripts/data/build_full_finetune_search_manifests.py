#!/usr/bin/env python3
"""Build the three replay-ratio pilot manifests for the full-fine-tune search.

Each manifest is self-contained (dataset_split "train"/"validation" columns,
consumable directly by scripts/training/train.py via load_training_rows()) at
~1,600 training occurrences -- the same per-trial pilot scale E008 used for
the LoRA rank/LR search (100 steps, batch 4, accumulation 4). Train rows mix
a deterministic slice of v5's openslr-train-v2 Sinhala pool with a
deterministic slice of the 1,111-clip English teacher-replay pool (see
prepare_teacher_replay_manifest.py) at the target ratio. Validation is a
fixed 200-row deterministic slice of v5's openslr-validation-v2.

This is the search's DATA axis; scripts/training/optuna_search_full_finetune.py
(not yet written) is the search LOOP that points at these three manifests and
varies learning_rate per trial.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SINHALA_TRAIN = "reports/dataset-audit/full-finetune-search-sinhala-pilot/manifest.parquet"
VALIDATION = "reports/dataset-audit/full-finetune-search-validation-pilot/manifest.parquet"
ENGLISH_REPLAY = "reports/dataset-audit/teacher-replay-english/manifest.parquet"
# All three sources above are self-contained (embedded audio via
# package_full_finetune_search_kaggle_inputs.py / prepare_teacher_replay_manifest.py),
# so the manifests this script writes are portable -- safe to upload to Kaggle,
# not just runnable against local files.

TOTAL_OCCURRENCES = 1600
VALIDATION_ROWS = 200
RATIOS = [0.10, 0.20, 0.30]


def main() -> None:
    sinhala = pd.read_parquet(SINHALA_TRAIN).sort_values("sample_id").reset_index(drop=True)
    english = pd.read_parquet(ENGLISH_REPLAY).sort_values("sample_id").reset_index(drop=True)
    validation = pd.read_parquet(VALIDATION).sort_values("sample_id").reset_index(drop=True)

    validation_slice = validation.head(VALIDATION_ROWS).copy()
    validation_slice["dataset_split"] = "validation"
    # drop columns not present in the other two sources so concat stays clean
    common_cols = None

    for ratio in RATIOS:
        english_count = round(TOTAL_OCCURRENCES * ratio)
        sinhala_count = TOTAL_OCCURRENCES - english_count
        assert english_count <= len(english), "not enough unique English replay clips"
        assert sinhala_count <= len(sinhala), "not enough Sinhala train rows"

        train_english = english.head(english_count).copy()
        train_english["dataset_split"] = "train"
        train_sinhala = sinhala.head(sinhala_count).copy()
        train_sinhala["dataset_split"] = "train"

        pieces = [train_sinhala, train_english, validation_slice]
        if common_cols is None:
            common_cols = set(pieces[0].columns)
            for piece in pieces[1:]:
                common_cols &= set(piece.columns)
            common_cols = sorted(common_cols)
        aligned = [piece[common_cols] for piece in pieces]
        combined = pd.concat(aligned, ignore_index=True)

        output = f"reports/dataset-audit/full-finetune-search/manifest-replay{int(ratio*100):02d}.parquet"
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), output, compression="zstd")
        print(
            f"{output}: {len(train_sinhala)} sinhala + {len(train_english)} english "
            f"({len(train_english) / (len(train_sinhala) + len(train_english)):.1%} replay) "
            f"+ {len(validation_slice)} validation"
        )


if __name__ == "__main__":
    main()
