"""Validation harness: do same-phrase command embeddings actually cluster?

This is step 1 of the speaker-enrolled voice-command work, and it is meant
to be a throwaway/standalone check - it does not import or depend on an
`app.streaming.embeddings` module (that comes in step 2, once this has
answered the yes/no question). It reuses the already-loaded streaming
Whisper checkpoint's *encoder only* (never the decoder):

    feature_extractor(audio) -> WhisperModel.encode(features) -> StorageView
    -> np.array(...) -> mean-pool over time -> L2-normalise

Usage (from the repo root, with the backend venv active):

    python -m scripts.validate_command_embeddings path/to/wav_dir

The directory must contain files named `{command_id}_{n}.wav`, e.g.
`delete_1.wav`, `delete_2.wav`, `stop_1.wav`, ... - several takes of each
command phrase, ideally from more than one speaker/session so "different
takes of the same phrase" is a meaningful test and not just "the same
audio file twice".

What it prints:
  - mean/min/max Manhattan-distance-derived similarity for same-phrase
    pairs, and for different-phrase pairs (this is the metric
    app.streaming.embeddings.best_match actually scores on - it beat
    cosine similarity's separation on the first real recording session
    tried here, Cohen's d 2.47 vs 2.27; use --csv to compare cosine,
    Euclidean and Pearson too if you want to re-check that on new data)
  - the separation gap and a suggested threshold
  - any command whose own within-phrase similarity looks unusually low
    (a bad candidate phrase - likely too short, or too acoustically
    similar to something else, or the recordings themselves are noisy)

If same-phrase similarity is not clearly above different-phrase
similarity, per the task this approach does not work as-is and building
enrollment storage/matching on top of it would be wasted effort - fix the
phrase set or recording quality first, or stop.
"""

import argparse
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

from app.core.config import settings


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MIN_CLIP_SECONDS = 0.3
FILENAME_RE = re.compile(r"^(?P<command_id>.+)_(?P<n>\d+)\.wav$", re.IGNORECASE)


@dataclass(frozen=True)
class Clip:
    command_id: str
    path: Path
    embedding: np.ndarray


def _load_as_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        target_length = int(duration * SAMPLE_RATE)
        original_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        audio = np.interp(target_positions, original_positions, audio).astype("float32")

    return audio


def _trim_silence(audio: np.ndarray, vad_model) -> np.ndarray:
    """Crop to [first speech start, last speech end] per the Silero VAD.

    Padding differences between clips (someone leaves half a second of
    silence before speaking, someone doesn't) are noise for embedding
    comparison, not signal - trim them out before embedding.
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


def _embed(audio: np.ndarray, model: WhisperModel) -> np.ndarray:
    """Encoder-only embedding: mean-pool the encoder hidden states, L2-normalise."""

    features = model.feature_extractor(audio)
    encoder_output = model.encode(features)
    hidden_states = np.array(encoder_output)  # (1, time, dim)

    pooled = hidden_states.mean(axis=1)[0]  # (dim,)
    norm = np.linalg.norm(pooled)

    return pooled / norm if norm > 0 else pooled


def _load_clips(wav_dir: Path, model: WhisperModel, vad_model) -> list[Clip]:
    clips: list[Clip] = []

    for path in sorted(wav_dir.glob("*.wav")):
        match = FILENAME_RE.match(path.name)
        if not match:
            logger.warning("Skipping %s - doesn't match {command_id}_{n}.wav", path.name)
            continue

        audio = _load_as_16k_mono(path)
        audio = _trim_silence(audio, vad_model)
        duration = len(audio) / SAMPLE_RATE

        if duration < MIN_CLIP_SECONDS:
            logger.warning(
                "Skipping %s - only %.2fs of speech after VAD trim (min %.2fs)",
                path.name,
                duration,
                MIN_CLIP_SECONDS,
            )
            continue

        embedding = _embed(audio, model)
        clips.append(
            Clip(command_id=match.group("command_id"), path=path, embedding=embedding)
        )

    return clips


def _pairwise_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Several ways to compare two embeddings, not just cosine.

    Since every embedding is already L2-normalised (see `_embed`),
    cosine similarity and the plain dot product are mathematically
    identical here - only one is reported. Euclidean/Manhattan distance
    and Pearson correlation are genuinely different measures and can
    disagree with cosine on individual pairs even though all four are
    highly correlated in aggregate for unit vectors.
    """

    cosine_similarity = float(np.dot(a, b))
    euclidean_distance = float(np.linalg.norm(a - b))
    manhattan_distance = float(np.sum(np.abs(a - b)))
    # Pearson correlation: cosine similarity computed on mean-centred
    # vectors - differs from plain cosine whenever the two vectors' means
    # (average activation across dimensions) differ.
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    pearson_correlation = float(np.dot(a_centered, b_centered) / denom) if denom > 0 else 0.0

    return {
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "manhattan_distance": manhattan_distance,
        "pearson_correlation": pearson_correlation,
    }


def _manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Same transform as app.streaming.embeddings.manhattan_similarity,
    duplicated here rather than imported - this script is deliberately
    standalone (see module docstring)."""

    dim = a.shape[-1]
    max_possible_distance = 2 * np.sqrt(dim)

    return 1.0 - float(np.sum(np.abs(a - b))) / max_possible_distance


def _write_csv(clips: list[Clip], csv_path: Path) -> None:
    """Every pairwise comparison, one row each - for spreadsheet/plotting
    use rather than just the printed summary. `manhattan_distance` is
    what the app actually matches on, via `manhattan_similarity` (see
    app.streaming.embeddings.best_match) - the other columns are for
    inspection/comparison, not used at runtime."""

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_a",
                "command_a",
                "file_b",
                "command_b",
                "same_phrase",
                "cosine_similarity",
                "euclidean_distance",
                "manhattan_distance",
                "pearson_correlation",
            ]
        )
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                a, b = clips[i], clips[j]
                metrics = _pairwise_metrics(a.embedding, b.embedding)
                writer.writerow(
                    [
                        a.path.name,
                        a.command_id,
                        b.path.name,
                        b.command_id,
                        a.command_id == b.command_id,
                        f"{metrics['cosine_similarity']:.6f}",
                        f"{metrics['euclidean_distance']:.6f}",
                        f"{metrics['manhattan_distance']:.6f}",
                        f"{metrics['pearson_correlation']:.6f}",
                    ]
                )

    print(f"\nWrote {len(clips) * (len(clips) - 1) // 2} pairwise rows to {csv_path}")


def _pairwise_report(clips: list[Clip]) -> None:
    if len(clips) < 2:
        print("Need at least 2 usable clips to compare anything.")
        return

    by_command: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_command[clip.command_id].append(clip)

    same_scores: list[float] = []
    diff_scores: list[float] = []
    same_scores_by_command: dict[str, list[float]] = defaultdict(list)

    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            score = _manhattan_similarity(clips[i].embedding, clips[j].embedding)
            if clips[i].command_id == clips[j].command_id:
                same_scores.append(score)
                same_scores_by_command[clips[i].command_id].append(score)
            else:
                diff_scores.append(score)

    print(f"\nLoaded {len(clips)} usable clips across {len(by_command)} commands:")
    for command_id, command_clips in sorted(by_command.items()):
        print(f"  {command_id}: {len(command_clips)} samples")

    if not same_scores:
        print(
            "\nNo command has 2+ samples - can't measure same-phrase similarity at all. "
            "Record at least 2 takes per command."
        )
        return

    if not diff_scores:
        print("\nOnly one command present - can't measure different-phrase separation.")
        return

    same = np.array(same_scores)
    diff = np.array(diff_scores)

    print("\nSame-phrase pairs   (n=%d): mean=%.4f  min=%.4f  max=%.4f"
          % (len(same), same.mean(), same.min(), same.max()))
    print("Different-phrase pairs (n=%d): mean=%.4f  min=%.4f  max=%.4f"
          % (len(diff), diff.mean(), diff.min(), diff.max()))

    mean_gap = same.mean() - diff.mean()
    worst_case_gap = same.min() - diff.max()
    midpoint_threshold = (same.mean() + diff.mean()) / 2
    # The largest-margin threshold, if one exists cleanly: halfway between
    # the weakest same-phrase pair and the strongest different-phrase pair.
    margin_threshold = (same.min() + diff.max()) / 2

    print(f"\nSeparation gap (mean_same - mean_diff): {mean_gap:+.4f}")
    print(f"Worst-case gap (min_same - max_diff):    {worst_case_gap:+.4f}")
    print(f"Suggested threshold (midpoint of means):  {midpoint_threshold:.4f}")
    print(f"Suggested threshold (max-margin):          {margin_threshold:.4f}")

    if worst_case_gap > 0:
        print(
            "\n=> Same-phrase pairs are ALWAYS more similar than different-phrase pairs. "
            "A single threshold cleanly separates them - the approach works on this data."
        )
    elif mean_gap > 0:
        print(
            "\n=> Same-phrase pairs are more similar ON AVERAGE, but the ranges overlap "
            "(some different-phrase pairs score higher than some same-phrase pairs). "
            "A single global threshold will misclassify some cases either way - check "
            "the flagged commands below before trusting this."
        )
    else:
        print(
            "\n=> Same-phrase similarity is NOT clearly above different-phrase similarity. "
            "Per the task, this means the approach does not work as-is on this data - "
            "stop and reconsider (different phrases, cleaner recordings, or more samples) "
            "rather than building enrollment storage on top of this."
        )

    print("\nPer-command within-phrase similarity (flag anything much below the others):")
    overall_same_mean = same.mean()
    for command_id in sorted(same_scores_by_command):
        scores = np.array(same_scores_by_command[command_id])
        flag = " <-- LOW, bad candidate phrase?" if scores.mean() < diff.mean() else ""
        print(
            f"  {command_id:12s} mean={scores.mean():.4f} min={scores.min():.4f} "
            f"max={scores.max():.4f} (n={len(scores)}){flag}"
        )
        if not flag and scores.mean() < overall_same_mean - 0.10:
            print(
                f"               (notably below the {overall_same_mean:.4f} average across "
                "all commands - worth a listen)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wav_dir", type=Path, help="Directory of {command_id}_{n}.wav recordings."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write every pairwise similarity to this CSV file.",
    )
    arguments = parser.parse_args()

    if not arguments.wav_dir.is_dir():
        raise SystemExit(f"No such directory: {arguments.wav_dir}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print(f"Loading Whisper checkpoint from {settings.streaming_model_ct2_path} ...")
    model = WhisperModel(
        settings.streaming_model_ct2_path,
        device="cpu",
        compute_type=settings.streaming_compute_type,
    )
    vad_model = load_silero_vad()

    clips = _load_clips(arguments.wav_dir, model, vad_model)
    _pairwise_report(clips)

    if arguments.csv:
        _write_csv(clips, arguments.csv)


if __name__ == "__main__":
    main()
