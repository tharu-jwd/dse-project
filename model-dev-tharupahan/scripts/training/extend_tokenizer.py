#!/usr/bin/env python3
"""Extend Whisper-small's tokenizer with dedicated Sinhala subword tokens.

Rationale (see docs/audits/e006-near-homophone-error-analysis.md and
docs/project/plan.md item 3): `openai/whisper-small`'s tokenizer has zero
dedicated Sinhala vocabulary -- every Sinhala Unicode codepoint falls back to
two raw UTF-8 byte-level tokens, and this project's single worst confusion
pair (le/lla) is one of the ones that gets no structural help from that
fallback. External precedent (ICASSP 2025, 8 Indic-script languages) found
250 new BPE tokens learned from language-specific data to be the optimal
point on a 125/250/500/1000 sweep, with random initialization for the new
embedding rows and no reported training instability.

Method: train a *fresh* tokenizer over this project's own Sinhala transcripts
using the base tokenizer's own `train_new_from_iterator` (same pre-tokenizer,
same byte-level alphabet, so any token string it produces that is genuinely
new represents a real learned multi-byte Sinhala merge, not an artifact of a
different tokenization scheme). Diff its vocabulary against the base
tokenizer's, keep the top N candidates in merge order (merges are learned in
roughly frequency order, so ID order is fine as a proxy), and append them to
the *original* tokenizer via `add_tokens()` -- this leaves every existing
token, ID, and merge untouched; the new tokens are purely additive.

Does not touch the model at all. `scripts/training/train.py` handles
`model.resize_token_embeddings()` when `extended_tokenizer_path` is set in
the training config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import WhisperTokenizer


def load_training_texts(manifest: Path) -> list[str]:
    table = pq.read_table(manifest, columns=["dataset_split", "text_canonical"])
    rows = table.to_pylist()
    texts = [
        row["text_canonical"]
        for row in rows
        if row.get("dataset_split") == "train" and row.get("text_canonical")
    ]
    if not texts:
        raise ValueError(f"no train-split text_canonical rows found in {manifest}")
    return texts


def select_new_tokens(
    base_tokenizer: WhisperTokenizer,
    texts: list[str],
    *,
    candidate_pool: int,
    target_new_tokens: int,
) -> list[str]:
    # train_new_from_iterator's vocab entries are expressed in the tokenizer's
    # internal byte-level alphabet (e.g. Sinhala 'ya' comes back as the
    # 3-character string "à¶º", not "ya" itself) -- that
    # alphabet is what the BPE merge algorithm operates on internally, but
    # add_tokens() matches literal, human-readable text against the *raw*
    # input before that remapping ever runs. Adding the byte-remapped form
    # directly is a silent no-op: it can never match anything in real text.
    # convert_tokens_to_string() reverses the remapping back to real
    # characters, which is what add_tokens() actually needs to match on.
    base_vocab = set(base_tokenizer.get_vocab().keys())
    trained = base_tokenizer.train_new_from_iterator(
        iter(texts), vocab_size=len(base_vocab) + candidate_pool
    )
    trained_vocab_by_id = sorted(trained.get_vocab().items(), key=lambda kv: kv[1])
    seen: set[str] = set()
    candidates: list[str] = []
    for token, _idx in trained_vocab_by_id:
        if token in base_vocab:
            continue
        decoded = trained.convert_tokens_to_string([token])
        if not decoded or decoded.isspace() or decoded in seen:
            continue
        seen.add(decoded)
        candidates.append(decoded)
    return candidates[:target_new_tokens]


def tokens_per_word(tokenizer: Any, texts: list[str]) -> float:
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    return total_tokens / total_words if total_words else 0.0


def fingerprint_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(file.name.encode("utf-8"))
            digest.update(file.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-new-tokens", type=int, default=250)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=2000,
        help="vocab_size budget for the throwaway trained tokenizer; must exceed "
        "target-new-tokens so there is room to filter out tokens the base "
        "tokenizer already has.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=2000,
        help="how many train-split rows to use for the before/after "
        "tokens-per-word report (separate random slice, not exhaustive).",
    )
    args = parser.parse_args()

    if args.candidate_pool <= args.target_new_tokens:
        raise ValueError("--candidate-pool must exceed --target-new-tokens")

    texts = load_training_texts(args.manifest)
    print(f"loaded {len(texts)} train-split transcripts from {args.manifest}")

    base_tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language="si", task="transcribe"
    )
    base_vocab_size = len(base_tokenizer)
    print(f"base tokenizer vocab size: {base_vocab_size}")

    new_tokens = select_new_tokens(
        base_tokenizer,
        texts,
        candidate_pool=args.candidate_pool,
        target_new_tokens=args.target_new_tokens,
    )
    print(f"selected {len(new_tokens)} new tokens (requested {args.target_new_tokens})")
    if len(new_tokens) < args.target_new_tokens:
        print(
            "warning: fewer novel tokens found than requested -- consider raising "
            "--candidate-pool"
        )

    extended_tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language="si", task="transcribe"
    )
    added = extended_tokenizer.add_tokens(new_tokens)
    print(f"add_tokens() actually added {added} tokens (duplicates are silently skipped)")
    extended_vocab_size = len(extended_tokenizer)

    eval_texts = texts[: args.eval_sample_size]
    before_tpw = tokens_per_word(base_tokenizer, eval_texts)
    after_tpw = tokens_per_word(extended_tokenizer, eval_texts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extended_tokenizer.save_pretrained(args.output_dir)
    tokenizer_sha256 = fingerprint_dir(args.output_dir)

    report = {
        "model_name": args.model_name,
        "manifest": str(args.manifest),
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": extended_vocab_size,
        "tokens_added": added,
        "candidate_pool": args.candidate_pool,
        "target_new_tokens": args.target_new_tokens,
        "train_texts_used_for_bpe_training": len(texts),
        "eval_sample_size": len(eval_texts),
        "tokens_per_word_before": before_tpw,
        "tokens_per_word_after": after_tpw,
        "tokens_per_word_reduction_pct": (
            100.0 * (before_tpw - after_tpw) / before_tpw if before_tpw else 0.0
        ),
        "new_tokens": new_tokens,
        "extended_tokenizer_dir_sha256": tokenizer_sha256,
    }
    report_path = args.output_dir / "extension-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"tokens/word before: {before_tpw:.3f}")
    print(f"tokens/word after:  {after_tpw:.3f}  ({report['tokens_per_word_reduction_pct']:.1f}% fewer)")
    print(f"saved extended tokenizer to {args.output_dir}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
