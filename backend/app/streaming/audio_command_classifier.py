"""English command classifier over the raw audio clip.

Uses the openWake-trained model (openWake/results/README.md): openWakeWord's
EAR embeddings feeding a fitted sklearn pipeline. The clip is prepared
exactly as in training (openWake/prepare_audio.py) - trim, then pad/crop to
3s. Loaded lazily, once, since importing openwakeword/librosa is slow.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app.core.config import settings

SR = 16000
CLIP_LEN = SR * 3

# The classifier's labels vs. the command ids the rest of the app uses.
_LABEL_TO_COMMAND_ID = {
    "option1": "option_1",
    "option2": "option_2",
    "option3": "option_3",
    "option4": "option_4",
}
_NONE_LABEL = "none"


@dataclass(frozen=True)
class AudioPrediction:
    command_id: str | None  # None when the label is "none"
    confidence: float


_lock = threading.Lock()
_model = None
_ear = None


def _load():
    global _model, _ear

    if _model is None:
        import joblib
        from openwakeword.utils import AudioFeatures

        _model = joblib.load(settings.voice_command_audio_model_source_path)
        _ear = AudioFeatures()

    return _model, _ear


def _prepare(audio: np.ndarray) -> np.ndarray:
    import librosa

    y, _ = librosa.effects.trim(audio.astype(np.float32), top_db=30)

    if len(y) >= CLIP_LEN:
        start = (len(y) - CLIP_LEN) // 2
        y = y[start : start + CLIP_LEN]
    else:
        pad = CLIP_LEN - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def classify(audio: np.ndarray) -> AudioPrediction:
    """Blocking - call via asyncio.to_thread."""

    clip = _prepare(audio)

    with _lock:
        model, ear = _load()
        features = ear.embed_clips(clip[None, :], batch_size=1)
        x = features.reshape(1, -1).astype(np.float32)
        probabilities = model.predict_proba(x)[0]

    index = int(np.argmax(probabilities))
    label = str(model.classes_[index])
    confidence = float(probabilities[index])

    if label == _NONE_LABEL:
        return AudioPrediction(None, confidence)

    return AudioPrediction(_LABEL_TO_COMMAND_ID.get(label, label), confidence)
