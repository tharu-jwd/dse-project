"""Splits data_train/<label>/*.wav into data_split/train/<label>/ and
data_split/val/<label>/, stratified per label so every class is represented
in both sets.
"""
import os
import random
import shutil

SRC = "data_train"
DST = "data_split"
VAL_FRACTION = 0.15
SEED = 42


def main():
    random.seed(SEED)
    for label in sorted(os.listdir(SRC)):
        src_folder = os.path.join(SRC, label)
        if not os.path.isdir(src_folder):
            continue
        files = [f for f in os.listdir(src_folder) if f.lower().endswith(".wav")]
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRACTION))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            out_dir = os.path.join(DST, split_name, label)
            os.makedirs(out_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(src_folder, f), os.path.join(out_dir, f))

        print(f"{label:10s} total={len(files):5d} train={len(train_files):5d} val={len(val_files):5d}")


if __name__ == "__main__":
    main()
