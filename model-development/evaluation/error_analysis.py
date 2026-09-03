"""Cluster and characterize what a fine-tuned Whisper model gets wrong on the
test set, using the per-sample prediction CSVs that `evaluate_finetuned.py`
and `evaluate_baselines.py` already write out (columns: reference,prediction
-- `evaluate_baselines.py`'s --output-csv has an extra leading `model` column,
both are accepted).

This does NOT re-run inference -- it only needs the CSVs you already have
(locally, or under gs://singen/whisper/finalData/... if that's where you
upload them). Run one of the eval scripts first if you don't have a
predictions CSV yet, e.g.:
    python3 evaluate_finetuned.py --model <checkpoint-dir> --output-dir eval_results/

What it produces per model (under --output-dir):
  <name>_errors_by_severity.csv   -- every wrong sample, worst WER first,
                                      with the reference/prediction/ops so you
                                      can read failures directly
  <name>_confusions.txt           -- most common word-level substitution
                                      pairs (ref -> hyp), and most commonly
                                      deleted / inserted words -- usually the
                                      fastest way to spot a systematic issue
                                      (e.g. a dropped honorific, a
                                      consistently misspelled loanword, a
                                      confused vowel-sign pair)
  <name>_clusters.txt             -- failed samples grouped by TF-IDF
                                      (character n-gram, so it works on
                                      Sinhala script without a tokenizer) +
                                      KMeans, so you can see *themes* in the
                                      failures (e.g. "numbers", "a speaker
                                      accent", "long sentences", "background
                                      noise transcripts") rather than reading
                                      every row one by one
  comparison_summary.csv          -- one row per model: WER/CER + how much of
                                      the error is substitutions vs deletions
                                      vs insertions, so you can tell *why*
                                      WER differs between models, not just
                                      that it does

Usage: point it at one or more predictions CSVs (one per model)
    python3 error_analysis.py \\
        --predictions eval_results/run6-lr3e-5-bs32_predictions.csv \\
        --predictions eval_results/run1-lora_predictions.csv \\
        --output-dir eval_results/error_analysis

A gs:// path works directly (pandas + gcsfs):
    python3 error_analysis.py --predictions gs://singen/whisper/finalData/eval_results/run6_predictions.csv
"""

import argparse
import os
from collections import Counter

import jiwer
import pandas as pd

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

# WER buckets for the severity CSV / cluster report -- not a metric, just a
# reading aid so you can jump straight to "completely wrong" rows.
SEVERITY_BINS = [
    (0.0, 0.0, "exact_match"),
    (0.0, 0.25, "minor"),
    (0.25, 0.5, "moderate"),
    (0.5, 1.0, "major"),
    (1.0, float("inf"), "severe"),  # WER > 100%: more inserted/substituted words than the reference has
]


def severity_label(wer):
    for lo, hi, label in SEVERITY_BINS:
        if lo <= wer <= hi if hi != float("inf") else wer > lo:
            return label
    return "severe"


def load_predictions(path):
    df = pd.read_csv(path)
    missing = {"reference", "prediction"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing} -- expected 'reference' and 'prediction' "
                          f"(the format evaluate_finetuned.py / evaluate_baselines.py write)")
    df["reference"] = df["reference"].fillna("").astype(str)
    df["prediction"] = df["prediction"].fillna("").astype(str)
    return df


def per_sample_ops(references, predictions):
    """jiwer.process_words gives per-sample alignment ops in one call --
    substitutions/deletions/insertions per row, plus overall WER."""
    norm_refs = [NORMALIZE(r) for r in references]
    norm_preds = [NORMALIZE(p) for p in predictions]
    # empty-after-normalization refs break jiwer's alignment; keep a
    # placeholder so the row still scores (as all-insertion) instead of crashing
    norm_refs = [r if r else "[empty]" for r in norm_refs]
    norm_preds = [p if p else "[empty]" for p in norm_preds]
    return jiwer.process_words(norm_refs, norm_preds), norm_refs, norm_preds


def analyze_model(name, df, output_dir, n_clusters, top_k):
    result, norm_refs, norm_preds = per_sample_ops(df["reference"].tolist(), df["prediction"].tolist())

    rows = []
    sub_pairs = Counter()
    deleted_words = Counter()
    inserted_words = Counter()
    total_sub = total_del = total_ins = total_hits = 0

    for i, (ref_words, hyp_words, alignment) in enumerate(zip(result.references, result.hypotheses, result.alignments)):
        n_sub = n_del = n_ins = n_hit = 0
        for chunk in alignment:
            if chunk.type == "substitute":
                n_sub += chunk.ref_end_idx - chunk.ref_start_idx
                for r_i, h_i in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                                     range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    sub_pairs[(ref_words[r_i], hyp_words[h_i])] += 1
            elif chunk.type == "delete":
                n_del += chunk.ref_end_idx - chunk.ref_start_idx
                for r_i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    deleted_words[ref_words[r_i]] += 1
            elif chunk.type == "insert":
                n_ins += chunk.hyp_end_idx - chunk.hyp_start_idx
                for h_i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    inserted_words[hyp_words[h_i]] += 1
            elif chunk.type == "equal":
                n_hit += chunk.ref_end_idx - chunk.ref_start_idx

        total_sub += n_sub
        total_del += n_del
        total_ins += n_ins
        total_hits += n_hit

        n_ref_words = max(len(ref_words), 1)
        sample_wer = (n_sub + n_del + n_ins) / n_ref_words
        rows.append({
            "reference": df["reference"].iloc[i],
            "prediction": df["prediction"].iloc[i],
            "wer": round(sample_wer, 4),
            "substitutions": n_sub,
            "deletions": n_del,
            "insertions": n_ins,
            "ref_len_words": len(ref_words),
            "severity": severity_label(sample_wer),
        })

    detail = pd.DataFrame(rows).sort_values("wer", ascending=False)
    wrong = detail[detail["wer"] > 0].copy()

    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, f"{name}_errors_by_severity.csv")
    detail.to_csv(detail_path, index=False)

    conf_path = os.path.join(output_dir, f"{name}_confusions.txt")
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(f"=== {name}: top {top_k} word substitutions (reference -> prediction) ===\n")
        for (ref_w, hyp_w), count in sub_pairs.most_common(top_k):
            f.write(f"  {count:5d}  {ref_w!r:20} -> {hyp_w!r}\n")
        f.write(f"\n=== {name}: top {top_k} deleted words (model dropped these) ===\n")
        for word, count in deleted_words.most_common(top_k):
            f.write(f"  {count:5d}  {word!r}\n")
        f.write(f"\n=== {name}: top {top_k} inserted words (model hallucinated these) ===\n")
        for word, count in inserted_words.most_common(top_k):
            f.write(f"  {count:5d}  {word!r}\n")
        f.write(f"\n=== {name}: severity distribution ===\n")
        for label in ["exact_match", "minor", "moderate", "major", "severe"]:
            n = (detail["severity"] == label).sum()
            f.write(f"  {label:12} {n:5d} samples ({100 * n / len(detail):.1f}%)\n")
    print(f"[{name}] wrote {detail_path} and {conf_path}")

    cluster_path = os.path.join(output_dir, f"{name}_clusters.txt")
    write_clusters(name, wrong, n_clusters, cluster_path)

    n_words_ref = total_hits + total_sub + total_del
    return {
        "model": name,
        "n_samples": len(detail),
        "n_wrong_samples": len(wrong),
        "wer_pct": 100 * (total_sub + total_del + total_ins) / max(n_words_ref, 1),
        "substitution_rate_pct": 100 * total_sub / max(n_words_ref, 1),
        "deletion_rate_pct": 100 * total_del / max(n_words_ref, 1),
        "insertion_rate_pct": 100 * total_ins / max(n_words_ref, 1),
    }


def write_clusters(name, wrong_df, n_clusters, out_path):
    """Group wrong samples by TF-IDF over character n-grams (script-agnostic,
    no Sinhala tokenizer needed) + KMeans, to surface *themes* in the
    failures (e.g. all the severe rows happen to be numbers, or all share a
    proper noun the model never saw in training)."""
    if len(wrong_df) < max(n_clusters, 2):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== {name}: not enough wrong samples ({len(wrong_df)}) to cluster ===\n")
        return

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = (wrong_df["reference"] + " " + wrong_df["prediction"]).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=5000)
    X = vectorizer.fit_transform(texts)

    k = min(n_clusters, len(wrong_df))
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    wrong_df = wrong_df.assign(cluster=labels)

    terms = vectorizer.get_feature_names_out()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== {name}: {len(wrong_df)} wrong samples grouped into {k} clusters ===\n")
        f.write("(clusters are by textual similarity of reference+prediction, not meaning -- \n"
                " use them to spot repeated failure patterns, e.g. same speaker/topic/word)\n\n")
        for c in range(k):
            sub = wrong_df[wrong_df["cluster"] == c].sort_values("wer", ascending=False)
            if sub.empty:
                continue
            center = km.cluster_centers_[c]
            top_terms = [terms[i] for i in center.argsort()[::-1][:8]]
            f.write(f"--- cluster {c} ({len(sub)} samples, avg WER {sub['wer'].mean():.2f}, "
                    f"top n-grams: {', '.join(top_terms)}) ---\n")
            for _, row in sub.head(5).iterrows():
                f.write(f"  WER={row['wer']:.2f}  ref:  {row['reference']}\n")
                f.write(f"              pred: {row['prediction']}\n")
            if len(sub) > 5:
                f.write(f"  ... and {len(sub) - 5} more in this cluster (see {name}_errors_by_severity.csv)\n")
            f.write("\n")
    print(f"[{name}] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", action="append", required=True, dest="predictions",
                         metavar="PATH",
                         help="a predictions CSV (reference,prediction[,model]) written by evaluate_finetuned.py "
                              "or evaluate_baselines.py; local path or gs://...; repeatable, one per model")
    parser.add_argument("--name", action="append", default=[], dest="names",
                         help="display name for each --predictions in order; defaults to the filename")
    parser.add_argument("--output-dir", default="error_analysis", help="where to write reports")
    parser.add_argument("--n-clusters", type=int, default=8, help="KMeans clusters for the failure-theme report")
    parser.add_argument("--top-k", type=int, default=25, help="how many confusion/deletion/insertion entries to list")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.predictions):
        parser.error("--name must be passed once per --predictions if used at all")

    summaries = []
    for i, path in enumerate(args.predictions):
        name = args.names[i] if args.names else os.path.splitext(os.path.basename(path.rstrip("/")))[0]
        print(f"\n=== Analyzing {name} ({path}) ===")
        df = load_predictions(path)
        if "model" in df.columns and df["model"].nunique() > 1:
            print(f"  note: {path} contains multiple models in the 'model' column; analyzing all rows together. "
                  f"Split it upstream if you need them separate.")
        summaries.append(analyze_model(name, df, args.output_dir, args.n_clusters, args.top_k))

    summary_df = pd.DataFrame(summaries).sort_values("wer_pct")
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n=== Comparison summary (sorted by WER) ===")
    print(summary_df.to_string(index=False))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
