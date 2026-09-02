"""Export real command-sample embeddings to JSON for visualization.

Reuses app.streaming.embeddings.embed_audio (the actual runtime encoder
path) against the real recordings in storage/voice_samples/ - the same
31 clips validate_command_embeddings.py and command_embedding_similarities.csv
were built from. Writes 768-dim embeddings plus a 2D PCA projection so a
plot can be drawn without re-running the model.

Usage (from backend/, with the venv active):
    python -m scripts.export_command_embeddings storage/voice_samples ../command_embeddings.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.streaming.embeddings import embed_audio
from app.streaming.inference import get_streaming_transcriber

SAMPLE_RATE = 16_000
FILENAME_RE = re.compile(r"^(?P<command_id>.+)_(?P<n>\d+)\.wav$", re.IGNORECASE)


def load_16k_mono(path: Path) -> np.ndarray:
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


def main() -> None:
    wav_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("storage/voice_samples")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../command_embeddings.json")

    model = get_streaming_transcriber()._model

    clips = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        match = FILENAME_RE.match(wav_path.name)
        if not match:
            continue
        audio = load_16k_mono(wav_path)
        embedding = embed_audio(model, audio)
        clips.append(
            {
                "command_id": match.group("command_id"),
                "file": wav_path.name,
                "embedding": embedding,
            }
        )
        print(f"embedded {wav_path.name} -> {match.group('command_id')}")

    matrix = np.stack([c["embedding"] for c in clips])
    pca2 = PCA(n_components=2, random_state=0)
    coords_2d = pca2.fit_transform(matrix)

    records = []
    for clip, xy in zip(clips, coords_2d):
        records.append(
            {
                "command_id": clip["command_id"],
                "file": clip["file"],
                "pca2": [float(xy[0]), float(xy[1])],
            }
        )

    output = {
        "points": records,
        "explained_variance_ratio": [float(v) for v in pca2.explained_variance_ratio_],
        "dim": int(matrix.shape[1]),
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"wrote {len(records)} embeddings to {out_path}")
    print(f"explained variance ratio (PC1, PC2): {pca2.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
