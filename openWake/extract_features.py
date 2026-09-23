"""
extract_features.py
Plan Step B6 / plan.md Section 4 & 8.2: run every clip in data_split/train
and data_split/val through openWakeWord's EAR (AudioFeatures) and save the
resulting embeddings + labels to disk, so train_classifier.py doesn't have
to redo this expensive step every time.

Requires: split_dataset.py to have already been run (creates data_split/).

Run with: python extract_features.py
Output:   features/train.npz, features/val.npz
          each containing X (N, 1536) float32 and y (N,) string labels
"""

import os
import numpy as np
from openwakeword.utils import AudioFeatures

from prepare_audio import load_clip, prepare, to_int16

SPLIT_DIR = "data_split"
OUT_DIR = "features"
BATCH_SIZE = 64


def list_labels(split_dir):
    train_dir = os.path.join(split_dir, "train")
    return sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )


def build_dataset(ear, split_dir, split_name, labels):
    folder = os.path.join(split_dir, split_name)
    clips, clip_labels = [], []

    for label in labels:
        label_dir = os.path.join(folder, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            y = load_clip(os.path.join(label_dir, fname))
            clips.append(prepare(y))
            clip_labels.append(label)

    audio = np.stack([to_int16(c) for c in clips])
    feats = ear.embed_clips(audio, batch_size=BATCH_SIZE)   # (N, 16, 96)
    X = feats.reshape(len(feats), -1).astype(np.float32)    # (N, 1536)
    y = np.array(clip_labels)
    return X, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = list_labels(SPLIT_DIR)
    print(f"Labels ({len(labels)}): {labels}")

    ear = AudioFeatures()

    for split_name in ("train", "val"):
        print(f"Extracting features for '{split_name}'...")
        X, y = build_dataset(ear, SPLIT_DIR, split_name, labels)
        out_path = os.path.join(OUT_DIR, f"{split_name}.npz")
        np.savez(out_path, X=X, y=y, labels=np.array(labels))
        print(f"  saved {out_path}: X={X.shape}, y={y.shape}")
        counts = {label: int((y == label).sum()) for label in labels}
        for label, count in counts.items():
            print(f"    {label:10s} {count}")


if __name__ == "__main__":
    main()
