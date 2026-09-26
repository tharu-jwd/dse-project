"""One-time prep: turn data_train/myRec/<label>/*.{mp4,wav,m4a} into
my_recordings/<label>/*.wav — the layout generate.py's generate_my_voice() expects.

- Converts every file to 16 kHz mono WAV via ffmpeg (handles WhatsApp .mp4 voice notes).
- Renames numeric folders (1,2,3,4) to option1..option4 to match COMMANDS labels.
- Balances classes: trims every label down to the smallest label's count, so the
  speed/pitch augmentation in generate.py multiplies them into an equal-sized set.
"""
import os
import random
import subprocess

SRC = "data_train/myRec"
DST = "my_recordings"
SEED = 42

LABEL_RENAME = {"1": "option1", "2": "option2", "3": "option3", "4": "option4"}


def convert_to_wav(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    random.seed(SEED)
    if not os.path.isdir(SRC):
        print(f"'{SRC}' not found.")
        return

    per_label_files = {}
    for raw_label in sorted(os.listdir(SRC)):
        folder = os.path.join(SRC, raw_label)
        if not os.path.isdir(folder):
            continue
        label = LABEL_RENAME.get(raw_label, raw_label)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".mp4", ".wav", ".m4a", ".mp3"))]
        per_label_files[label] = [(folder, f) for f in files]

    min_count = min(len(v) for v in per_label_files.values())
    print("Counts before balancing:")
    for label, files in per_label_files.items():
        print(f"  {label:10s} {len(files)}")
    print(f"Balancing every label down to {min_count} recordings.\n")

    for label, files in per_label_files.items():
        random.shuffle(files)
        kept = files[:min_count]
        out_dir = os.path.join(DST, label)
        os.makedirs(out_dir, exist_ok=True)
        for i, (folder, fname) in enumerate(kept, 1):
            src_path = os.path.join(folder, fname)
            dst_path = os.path.join(out_dir, f"{label}_{i:02d}.wav")
            try:
                convert_to_wav(src_path, dst_path)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {src_path} ({e})")
        print(f"  {label:10s} -> {len(kept)} wav files in {out_dir}")

    print("\nDone. Now you can run: python generate.py")


if __name__ == "__main__":
    main()
