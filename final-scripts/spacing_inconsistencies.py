"""Find spacing inconsistencies in the Sinhala ASR reference transcripts,
i.e. compound words that are sometimes written as one token and sometimes as
two tokens.

Two kinds of check:

  1. train-vs-predictions (original check): a word seen as one token in
     train_refs.txt but produced as two tokens (a bigram) by the model in the
     eval predictions CSVs (run*.csv under eval_results/). This tells you
     "the model can't decide how to space this word" -- often because train
     itself is inconsistent.

  2. within-split (new): for each of train/test/validation independently,
     a word that appears as one token *and* as a split bigram inside the
     same split's own reference transcripts. This tells you the split's
     ground truth itself is inconsistently spaced, independent of any model.

Usage:
    python3 spacing_inconsistencies.py
Expects, under eval_results/: train_refs.txt, test_refs.txt,
validation_refs.txt (one reference sentence per line) and run*.csv
(reference,prediction columns) for the model-vs-train check.
"""
import glob
import os
import re
from collections import Counter

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
WORD_RE = re.compile(r"\S+")


def word_counts(lines):
    c = Counter()
    for line in lines:
        c.update(WORD_RE.findall(line))
    return c


def bigram_counts(lines):
    """Count adjacent-word pairs joined with no space, i.e. what the token
    would look like if the two words were fused into one."""
    c = Counter()
    for line in lines:
        words = WORD_RE.findall(line)
        for a, b in zip(words, words[1:]):
            c[(a, b)] += 1
    return c


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def train_vs_predictions_section(train_lines):
    pred_files = [os.path.join(EVAL_DIR, "run1_full.csv")]
    pred_lines = []
    for path in pred_files:
        import csv
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("prediction"):
                    pred_lines.append(row["prediction"])

    train_words = word_counts(train_lines)
    pred_bigrams = bigram_counts(pred_lines)

    entries = []
    for (a, b), n_pred in pred_bigrams.items():
        fused = a + b
        if fused in train_words:
            entries.append((fused, train_words[fused], f"{a} {b}", n_pred))
    entries.sort(key=lambda e: e[3], reverse=True)

    lines = []
    lines.append(f"total distinct inconsistent bigrams: {len(entries)}")
    lines.append(f"total occurrences: {sum(e[3] for e in entries)}")
    lines.append("")
    for fused, n_train, split, n_pred in entries:
        lines.append(f'{fused}\t(seen {n_train}x in train as one word)\tpredicted as: "{split}"\t({n_pred}x in predictions)')
    return "\n".join(lines)


def within_split_section(name, lines):
    """Words in `lines` that appear both fused (one token) and split (two
    adjacent tokens) somewhere in the same set of lines."""
    words = word_counts(lines)
    bigrams = bigram_counts(lines)

    entries = []
    for (a, b), n_split in bigrams.items():
        if len(a) < 2 or len(b) < 2:
            continue  # single-character pieces are usually coincidental fusions, not real compounds
        fused = a + b
        if fused in words:
            entries.append((fused, words[fused], f"{a} {b}", n_split))
    entries.sort(key=lambda e: e[3], reverse=True)

    out = []
    out.append(f"=== {name}: internal spacing inconsistencies (fused vs split within {name} itself) ===")
    out.append(f"total distinct inconsistent bigrams: {len(entries)}")
    out.append(f"total occurrences (fused + split): {sum(e[1] + e[3] for e in entries)}")
    out.append("")
    for fused, n_fused, split, n_split in entries:
        out.append(f'{fused}\t(seen {n_fused}x as one word)\talso seen as: "{split}"\t({n_split}x split in {name})')
    return "\n".join(out)


def main():
    train_lines = read_lines(os.path.join(EVAL_DIR, "train_refs.txt"))
    test_lines = read_lines(os.path.join(EVAL_DIR, "test_refs.txt"))
    val_lines = read_lines(os.path.join(EVAL_DIR, "validation_refs.txt"))

    sections = []
    sections.append("### SECTION 1: train words split by the model in predictions (run1_full.csv) ###\n\n"
                     + train_vs_predictions_section(train_lines))
    sections.append(within_split_section("train", train_lines))
    sections.append(within_split_section("test", test_lines))
    sections.append(within_split_section("validation", val_lines))

    out_path = os.path.join(EVAL_DIR, "spacing_inconsistencies.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(sections) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
