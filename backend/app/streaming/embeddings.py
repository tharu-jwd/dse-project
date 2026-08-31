"""Speaker-audio embeddings for voice-command matching - a second,
sound-based opinion alongside the text-based fuzzy matching in
`app.streaming.commands`.

Deliberately independent of that module: nothing here knows what a
command *is*, only "audio in -> vector out" and "given a bank of labeled
vectors, what does this new vector match best". The label is just an
arbitrary string chosen by the caller (a command id, in practice). The
command vocabulary in `commands.py` can be edited, extended, or
reordered at any time without touching this file - only what's already
been enrolled into a bank (built elsewhere, in a later step) would need
re-recording if a phrase itself changes.

Reuses the already-loaded faster-whisper encoder from
`StreamingTranscriber` - never loads a second model, never runs the
decoder:

    feature_extractor(audio) -> WhisperModel.encode(features)
    -> ctranslate2.StorageView -> np.array(...) -> mean-pool over time
    -> L2-normalise

Verified against the installed faster-whisper (1.2.1): `WhisperModel`
exposes `.feature_extractor` and `.encode()` directly, so no fallback to
a separately-loaded HF Whisper encoder was needed. See
`scripts/validate_command_embeddings.py`, which used this exact
sequence to confirm same-phrase recordings cluster before this module
was written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

from app.core.config import settings


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class ClipTooShortError(ValueError):
    """Raised when a clip has too little speech (after VAD trimming) to
    embed reliably."""


# ---------------------------------------------------------------------------
# Pure vector maths - no model involved, unit-testable with plain arrays.
# ---------------------------------------------------------------------------


def mean_pool(hidden_states: np.ndarray) -> np.ndarray:
    """Collapse a (..., time, dim) array to a fixed-length (..., dim) vector.

    Whisper's encoder emits one hidden vector per time frame; a clip's
    embedding is just their average, so clips of different lengths still
    produce comparable fixed-size vectors.
    """

    return hidden_states.mean(axis=-2)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so cosine similarity becomes a plain dot product."""

    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance, rescaled to a bounded "higher is better" score
    with roughly the same 0-1 feel as cosine similarity.

    `scripts/validate_command_embeddings.py` found Manhattan distance
    separates same-command from different-command pairs slightly better
    than cosine on real recordings (Cohen's d 2.47 vs 2.27) - this is
    what `best_match` actually scores on.

    The rescale isn't an arbitrary magic number: for two L2-normalised
    vectors of dimension `dim`, Manhattan distance is bounded above by
    `2 * sqrt(dim)` (Cauchy-Schwarz applied to the L2 distance, which is
    itself at most 2 for unit vectors). Dividing by that bound and
    subtracting from 1 gives an unarbitrary 0-1-ish scale that doesn't
    need re-deriving if the embedding dimension ever changes - only the
    *threshold* on this score is empirically tuned, in config.py, from
    `scripts/validate_command_embeddings.py`'s output.
    """

    dim = a.shape[-1]
    max_possible_distance = 2 * np.sqrt(dim)

    return 1.0 - manhattan_distance(a, b) / max_possible_distance


@dataclass(frozen=True)
class EmbeddingMatch:
    label: str
    score: float


def best_match(
    query: np.ndarray,
    bank: dict[str, list[np.ndarray]],
    *,
    threshold: float | None = None,
) -> EmbeddingMatch | None:
    """Score `query` against every embedding in `bank`, grouped by label.

    A label's score is its single best-matching sample, not the mean of
    its samples - one clean enrolled take should be enough to recognise
    a command even if another sample for that same label was noisy.

    `bank` maps an arbitrary label to the list of embeddings stored for
    it. This function has no idea what the labels mean, where the bank
    came from, or how many labels exist - that's entirely the caller's
    concern (today: command ids from an enrollment store).

    Returns `None` - never a low-confidence guess - when the bank is
    empty or nothing clears the threshold.
    """

    threshold = (
        settings.voice_embedding_similarity_threshold if threshold is None else threshold
    )

    best_label: str | None = None
    best_score = -1.0

    for label, samples in bank.items():
        for sample in samples:
            score = manhattan_similarity(query, sample)
            if score > best_score:
                best_score = score
                best_label = label

    if best_label is None or best_score < threshold:
        return None

    return EmbeddingMatch(label=best_label, score=best_score)


# ---------------------------------------------------------------------------
# Model-backed embedding extraction.
# ---------------------------------------------------------------------------

_vad_model = None


def _get_vad_model():
    global _vad_model

    if _vad_model is None:
        _vad_model = load_silero_vad()

    return _vad_model


def _trim_silence(audio: np.ndarray, vad_model) -> np.ndarray:
    """Crop to [first speech start, last speech end].

    Padding differences between clips (someone leaves half a second of
    silence before speaking, someone doesn't) are noise for embedding
    comparison, not signal - trimmed out before embedding, same as in
    the step-1 validation harness.
    """

    if audio.size == 0:
        return audio

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio), vad_model, sampling_rate=SAMPLE_RATE, return_seconds=True
    )

    if not timestamps:
        return audio

    start_sample = int(timestamps[0]["start"] * SAMPLE_RATE)
    end_sample = int(timestamps[-1]["end"] * SAMPLE_RATE)

    return audio[start_sample:end_sample]


def embed_audio(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    min_duration_seconds: float | None = None,
) -> np.ndarray:
    """Encoder-only embedding for one clip of 16kHz mono float32 audio.

    `model` is the faster-whisper `WhisperModel` already loaded by
    `StreamingTranscriber` - pass its instance in, never construct a
    second one here.
    """

    min_duration_seconds = (
        settings.voice_embedding_min_clip_seconds
        if min_duration_seconds is None
        else min_duration_seconds
    )

    trimmed = _trim_silence(audio, _get_vad_model())
    duration = trimmed.size / SAMPLE_RATE

    if duration < min_duration_seconds:
        raise ClipTooShortError(
            f"Clip has only {duration:.2f}s of speech after trimming "
            f"(minimum {min_duration_seconds:.2f}s)."
        )

    features = model.feature_extractor(trimmed)
    encoder_output = model.encode(features)
    hidden_states = np.array(encoder_output)[0]  # (time, dim), batch dim dropped

    return l2_normalize(mean_pool(hidden_states))
