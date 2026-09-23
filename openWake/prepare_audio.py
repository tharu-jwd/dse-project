"""
prepare_audio.py
Shared audio-preparation code (README_Command_Classifier_Plan.md, Step B4).

Used by BOTH extract_features.py (training) and app.py (runtime), so that
clips are prepared identically in training and in the live app.

Clip length: chosen from the actual data_train/*.wav durations
(max observed ~3.7s, p95 ~2.6s), so 2s (the README example's default) is
too tight for multi-word commands like "number one" or "cancel this".
CLIP_LEN_SEC = 3.0s gives headroom without wasting too much padding on the
short one-word commands.
"""

import librosa
import numpy as np

SR = 16000               # openWakeWord's EAR needs 16 kHz mono
CLIP_LEN_SEC = 3.0
CLIP_LEN = int(SR * CLIP_LEN_SEC)   # 48,000 samples


def load_clip(path):
    """Load any audio file as 16 kHz mono float32 in [-1, 1]."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def prepare(y):
    """Trim silence, then pad/crop to exactly CLIP_LEN samples, centered."""
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) >= CLIP_LEN:
        start = (len(y) - CLIP_LEN) // 2
        return y[start:start + CLIP_LEN]
    pad = CLIP_LEN - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def to_int16(y):
    """Convert float32 [-1, 1] audio to the int16 PCM the EAR expects."""
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)
