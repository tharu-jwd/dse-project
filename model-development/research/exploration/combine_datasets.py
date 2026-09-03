"""
Combine the three Sinhala ASR corpora (Lingalingeswaran, YouTube, BizBrains) into a
single DataFrame with a common schema, ready for batch preprocessing.

Row order is block-wise per dataset (Linga rows first, then YouTube, then BizBrains),
so a fixed contiguous index range identifies each source — e.g. if Linga has 4 rows
after cleaning, rows 0-3 are Linga and row 4 onward is YouTube. `source_dataset` is
also kept as an explicit column so later train/val/test splitting can sample a fixed
percentage from each source without relying on index math.

Only `audio` and `text` are required for fine-tuning; `duration`, `text_len`, and the
VAD-derived silence features are carried along purely to make later preprocessing
(filtering, stratified sampling, QA) easier — they are not model inputs.
"""

import io

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torch.hub
from pathlib import Path
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).parent

# Rows confirmed bad in youtube.ipynb section 8: duplicate "not suitable for
# transcription" placeholders (258, 1807, 3555), a duplicate audio/transcript pair
# (2095, keeping 1497), and the one clip Silero VAD found no speech in at all (3306).
YOUTUBE_DROP_IDS = [258, 1807, 3555, 2095, 3306]


def get_bytes(cell):
    if isinstance(cell, dict):
        return cell["bytes"]
    if isinstance(cell, (bytes, bytearray, np.ndarray)):
        return bytes(cell)
    raise TypeError(type(cell))


def get_duration(raw_bytes):
    try:
        return sf.info(io.BytesIO(raw_bytes)).duration
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Load each corpus and normalise to a common schema:
#    audio (raw WAV bytes), text, source_dataset, orig_split
# ---------------------------------------------------------------------------

def load_linga():
    base = BASE_DIR / "lingaData/data"
    files = sorted(base.glob("train-*-of-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    out = pd.DataFrame({
        "audio": df["audio_path"].apply(get_bytes),
        "text": df["transcription"],
        "source_dataset": "linga",
        "orig_split": None,
    })
    return out


def load_youtube():
    base = BASE_DIR / "youtubeData/data"
    files = {
        "train": sorted(base.glob("train-*-of-*.parquet")),
        "validation": sorted(base.glob("validation-*-of-*.parquet")),
        "test": sorted(base.glob("test-*-of-*.parquet")),
    }
    df = pd.concat(
        [pd.read_parquet(f).assign(split=name) for name, fs in files.items() for f in fs],
        ignore_index=True,
    )
    df = df.drop(index=YOUTUBE_DROP_IDS).reset_index(drop=True)

    out = pd.DataFrame({
        "audio": df["audio"].apply(get_bytes),
        "text": df["text"],
        "source_dataset": "youtube",
        "orig_split": df["split"],
    })
    return out


def load_bizbrains():
    base = BASE_DIR / "bizbrainsData/data"
    files = {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
    df = pd.concat(
        [pd.read_parquet(base / f).assign(split=name) for name, f in files.items()],
        ignore_index=True,
    )

    out = pd.DataFrame({
        "audio": df["audio"].apply(get_bytes),
        "text": df["text"],
        "source_dataset": "bizbrains",
        "orig_split": df["split"],
    })
    return out


# ---------------------------------------------------------------------------
# 2. VAD-based silence features (Silero VAD, not the energy-threshold method)
# ---------------------------------------------------------------------------

def load_vad():
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils[0]  # get_speech_timestamps


def vad_features(raw_bytes, model, get_speech_timestamps):
    x, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr

    wav = torch.from_numpy(x)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    ts = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)

    if not ts:
        return dict(vad_silence_frac=1.0, vad_lead_sil_s=dur, vad_trail_sil_s=dur,
                    vad_speech_s=0.0, vad_n_segments=0)

    speech_s = sum(t["end"] - t["start"] for t in ts)
    return dict(
        vad_silence_frac=max(0.0, 1 - speech_s / dur),
        vad_lead_sil_s=ts[0]["start"],
        vad_trail_sil_s=max(0.0, dur - ts[-1]["end"]),
        vad_speech_s=speech_s,
        vad_n_segments=len(ts),
    )


# ---------------------------------------------------------------------------
# 3. Assemble
# ---------------------------------------------------------------------------

def main():
    print("Loading corpora...")
    linga = load_linga()
    youtube = load_youtube()
    bizbrains = load_bizbrains()

    for name, d in [("linga", linga), ("youtube", youtube), ("bizbrains", bizbrains)]:
        print(f"  {name}: {len(d)} rows")

    # Block-wise concat: Linga first, then YouTube, then BizBrains. reset_index gives
    # each source a fixed contiguous range in the combined table.
    combined = pd.concat([linga, youtube, bizbrains], ignore_index=True)
    print(f"\nCombined: {combined.shape}")
    print(combined.groupby("source_dataset").size())

    print("\nComputing duration and text length...")
    combined["duration"] = [get_duration(b) for b in tqdm(combined["audio"], desc="duration")]
    combined["text_len"] = combined["text"].str.len()

    print("\nComputing VAD silence features (this loads Silero VAD via torch.hub)...")
    model, get_speech_timestamps = load_vad()
    vad_rows = [
        vad_features(b, model, get_speech_timestamps)
        for b in tqdm(combined["audio"], desc="VAD")
    ]
    combined = combined.join(pd.DataFrame(vad_rows))

    out_path = BASE_DIR / "combinedData1.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved combined dataset to {out_path}")
    print(combined.dtypes)


if __name__ == "__main__":
    main()
