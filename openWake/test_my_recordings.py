"""
test_my_recordings.py
Held-out real-voice test (plan Step 8.7): unlike my_recordings/, these clips
are NEVER augmented or mixed into data_train/ — they only exist to measure
real-world performance of the already-trained models.

One script does all three stages:
  1. PREPARE  — convert raw recordings in data_train/myRec_test/<label|1-4>/*
               to 16kHz mono WAV, save as data_split/test/<label>/*.wav
               (no speed/pitch augmentation — this is a test set, not train data)
  2. EXTRACT  — run every data_split/test clip through the EAR, save the
               embeddings to features/test.npz
  3. EVALUATE — load every trained model in results/*/model.joblib, predict
               on features/test.npz, and save a held-out report next to each
               model's existing validation results

Put your held-out recordings in:
    data_train/myRec_test/<label>/*.{wav,mp3,m4a,mp4}
    data_train/myRec_test/1/  ... /4/    (renamed to option1..option4)

Run with: python test_my_recordings.py
"""

import os
import glob
import json
import subprocess

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from openwakeword.utils import AudioFeatures

from prepare_audio import load_clip, prepare, to_int16

RAW_TEST_SRC = "my_recordings/"
TEST_SPLIT_DIR = "data_split/test"
FEATURES_OUT = "features/test.npz"
RESULTS_DIR = "results"
LABEL_RENAME = {"1": "option1", "2": "option2", "3": "option3", "4": "option4"}


# ---------------------------------------------------------------------------
# Stage 1: PREPARE — convert raw held-out recordings to data_split/test/
# ---------------------------------------------------------------------------

def convert_to_wav(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def prepare_test_set():
    if not os.path.isdir(RAW_TEST_SRC):
        raise SystemExit(
            f"'{RAW_TEST_SRC}' not found.\n"
            f"Put held-out recordings in {RAW_TEST_SRC}/<label>/*.{{wav,mp3,m4a,mp4}} "
            f"(use 1..4 for option1..option4) and re-run."
        )

    print(f"=== Stage 1: preparing test set from {RAW_TEST_SRC} ===")
    for raw_label in sorted(os.listdir(RAW_TEST_SRC)):
        folder = os.path.join(RAW_TEST_SRC, raw_label)
        if not os.path.isdir(folder):
            continue
        label = LABEL_RENAME.get(raw_label, raw_label)
        out_dir = os.path.join(TEST_SPLIT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".wav", ".mp3", ".m4a", ".mp4"))]
        for i, fname in enumerate(sorted(files), 1):
            src_path = os.path.join(folder, fname)
            dst_path = os.path.join(out_dir, f"{label}_test_{i:02d}.wav")
            try:
                convert_to_wav(src_path, dst_path)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {src_path} ({e})")
        print(f"  {label:10s} -> {len(files)} wav files in {out_dir}")


# ---------------------------------------------------------------------------
# Stage 2: EXTRACT — embed data_split/test/ through the EAR
# ---------------------------------------------------------------------------

def extract_test_features():
    print("\n=== Stage 2: extracting features for 'test' ===")
    labels = sorted(
        d for d in os.listdir(TEST_SPLIT_DIR)
        if os.path.isdir(os.path.join(TEST_SPLIT_DIR, d))
    )
    print(f"Labels ({len(labels)}): {labels}")

    ear = AudioFeatures()
    clips, clip_labels = [], []
    for label in labels:
        label_dir = os.path.join(TEST_SPLIT_DIR, label)
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            y = load_clip(os.path.join(label_dir, fname))
            clips.append(prepare(y))
            clip_labels.append(label)

    audio = np.stack([to_int16(c) for c in clips])
    feats = ear.embed_clips(audio, batch_size=64)
    X = feats.reshape(len(feats), -1).astype(np.float32)
    y = np.array(clip_labels)

    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
    np.savez(FEATURES_OUT, X=X, y=y, labels=np.array(labels))
    print(f"  saved {FEATURES_OUT}: X={X.shape}, y={y.shape}")
    return X, y, labels


# ---------------------------------------------------------------------------
# Stage 3: EVALUATE — run every trained model on the held-out test set
# ---------------------------------------------------------------------------

def evaluate_all_models(X_test, y_test, labels):
    print("\n=== Stage 3: evaluating trained models on held-out test set ===")
    model_paths = sorted(glob.glob(f"{RESULTS_DIR}/*/model.joblib"))
    if not model_paths:
        raise SystemExit(f"No trained models found under {RESULTS_DIR}/*/model.joblib")

    for model_path in model_paths:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_dir)
        out_dir = os.path.join(model_dir, "held_out_test")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- {model_name} ---")
        brain = joblib.load(model_path)
        pred = brain.predict(X_test)

        report_text = classification_report(y_test, pred, labels=labels)
        print(report_text)

        cm = confusion_matrix(y_test, pred, labels=labels)
        print("confusion_matrix (held-out test):")
        print("labels order:", labels)
        print(cm)

        wrong_from_none = total_none = None
        if "none" in labels:
            wrong_from_none = int(np.sum((y_test == "none") & (pred != "none")))
            total_none = int(np.sum(y_test == "none"))
            print(f"False-accepts from 'none': {wrong_from_none}/{total_none}")

        with open(f"{out_dir}/classification_report.txt", "w") as f:
            f.write(report_text)

        with open(f"{out_dir}/confusion_matrix.csv", "w") as f:
            f.write("," + ",".join(labels) + "\n")
            for label, row in zip(labels, cm):
                f.write(label + "," + ",".join(str(v) for v in row) + "\n")

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {model_name} (held-out test)")
        thresh = cm.max() / 2 if cm.max() else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center",
                             color="white" if v > thresh else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
        plt.close(fig)

        metrics = {
            "model": model_name,
            "set": "held_out_test (real, never-augmented recordings)",
            "accuracy": float(np.mean(pred == y_test)),
            "macro_f1": float(f1_score(y_test, pred, labels=labels, average="macro")),
            "weighted_f1": float(f1_score(y_test, pred, labels=labels, average="weighted")),
            "false_accepts_from_none": wrong_from_none,
            "total_none": total_none,
            "labels": labels,
            "n_test_clips": int(len(y_test)),
        }
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"Saved held-out results to {out_dir}/")


def main():
    prepare_test_set()
    X_test, y_test, labels = extract_test_features()
    evaluate_all_models(X_test, y_test, labels)


if __name__ == "__main__":
    main()
