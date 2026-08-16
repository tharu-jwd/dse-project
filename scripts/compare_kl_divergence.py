"""Compare `final_collection_qa.parquet` and `final_dataset_openslr_qa.parquet`
by KL divergence (plus Jensen-Shannon divergence) along two empirical distributions:

  1. Character frequency -- the distribution of individual Unicode characters across
     all text. Captures whether the two corpora share the same linguistic profile
     (script mix, punctuation, code-mixed Latin characters) or diverge.
  2. Audio duration -- binned histogram of clip length in seconds. Captures whether
     the two corpora have similar utterance-length profiles; a large mismatch here
     is itself a form of domain shift, relevant if you're combining them for
     training or treating one as held-out against the other.

KL divergence is asymmetric and undefined wherever Q has zero probability where P is
nonzero, so both directions -- KL(P||Q) and KL(Q||P) -- are reported, using additive
(Laplace-style) smoothing over the full symbol/bin support to avoid -inf/NaN from
support mismatches. Jensen-Shannon divergence (symmetric, bounded in [0, ln 2]) is
reported alongside as a more numerically robust companion metric.

Memory handling: `final_dataset_openslr_qa.parquet` embeds ~15GB of audio, so
duration is read via `soundfile.info()` on each row's bytes (WAV header only, not a
full decode) while streaming row-groups -- audio bytes are never held beyond the
batch they arrive in. `final_collection_qa.parquet` (4.9k rows) is small enough
to load whole.

Usage (from repo root):
    python3 dse-project/scripts/compare_kl_divergence.py
    python3 dse-project/scripts/compare_kl_divergence.py --top-n 20
"""

import argparse
import io
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

DEFAULT_COLLECTION = "dse-project/model-development/data/final_dataset/final_collection_qa.parquet"
DEFAULT_OPENSLR = "dse-project/model-development/data/final_dataset/final_dataset_openslr_qa.parquet"
LABEL_A, LABEL_B = "collection", "openslr"

DURATION_BIN_WIDTH = 0.5  # seconds
DURATION_MAX = 30.0       # clips longer than this collapse into one overflow bin
EPS_SMOOTHING = 1e-4      # Laplace-style smoothing mass added per symbol/bin


def kl_divergence(p, q):
    """KL(P || Q) in nats, both already-normalized 1D arrays over the same support."""
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def smoothed_dists(counts_a, counts_b):
    """Turn two Counters over the same symbol space into smoothed probability
    vectors aligned over their union of keys."""
    keys = sorted(set(counts_a) | set(counts_b), key=lambda k: (k is None, k))
    a = np.array([counts_a.get(k, 0) for k in keys], dtype=np.float64) + EPS_SMOOTHING
    b = np.array([counts_b.get(k, 0) for k in keys], dtype=np.float64) + EPS_SMOOTHING
    a /= a.sum()
    b /= b.sum()
    return keys, a, b


def report(name, counts_a, counts_b, top_n):
    keys, p, q = smoothed_dists(counts_a, counts_b)
    kl_pq, kl_qp, js = kl_divergence(p, q), kl_divergence(q, p), js_divergence(p, q)

    print(f"\n--- {name} ---")
    print(f"support size: {len(keys)}")
    print(f"KL({LABEL_A} || {LABEL_B}) = {kl_pq:.4f} nats")
    print(f"KL({LABEL_B} || {LABEL_A}) = {kl_qp:.4f} nats")
    print(f"Jensen-Shannon divergence   = {js:.4f} nats  (symmetric, max = ln(2) = {np.log(2):.4f})")

    contrib = p * np.log(p / q)
    order = np.argsort(-contrib)[:top_n]
    print(f"top {top_n} symbols/bins driving KL({LABEL_A} || {LABEL_B}):")
    for i in order:
        print(f"  {keys[i]!r:>8}  p={p[i]:.5f}  q={q[i]:.5f}  contrib={contrib[i]:.5f}")


def char_counts(texts):
    c = Counter()
    for t in texts:
        c.update(t)
    return c


def duration_bin(seconds):
    seconds = min(seconds, DURATION_MAX)
    return round((seconds // DURATION_BIN_WIDTH) * DURATION_BIN_WIDTH, 2)


def collection_stats(path):
    df = pd.read_parquet(path, columns=["text", "audio"])
    chars = char_counts(df["text"])
    durs = Counter()
    for b in df["audio"]:
        try:
            info = sf.info(io.BytesIO(b))
            durs[duration_bin(info.frames / info.samplerate)] += 1
        except Exception:
            continue
    return chars, durs


def openslr_stats(path, batch_size):
    chars, durs = Counter(), Counter()
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    seen = 0
    for batch in pf.iter_batches(columns=["text", "audio"], batch_size=batch_size):
        texts = batch.column("text").to_pylist()
        audios = batch.column("audio").to_pylist()
        chars.update(char_counts(texts))
        for b in audios:
            try:
                info = sf.info(io.BytesIO(b))
                durs[duration_bin(info.frames / info.samplerate)] += 1
            except Exception:
                continue
        seen += len(texts)
        print(f"  processed {seen}/{total} OpenSLR rows", end="\r")
    print()
    return chars, durs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--openslr", default=DEFAULT_OPENSLR)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--top-n", type=int, default=10, help="top divergence-contributing symbols/bins to print")
    args = parser.parse_args()

    print(f"Loading {LABEL_A} from {args.collection} ...")
    coll_chars, coll_durs = collection_stats(args.collection)

    print(f"Streaming {LABEL_B} from {args.openslr} ...")
    osl_chars, osl_durs = openslr_stats(args.openslr, args.batch_size)

    report("Character distribution", coll_chars, osl_chars, args.top_n)
    report("Duration-bin distribution (seconds)", coll_durs, osl_durs, args.top_n)


if __name__ == "__main__":
    main()
