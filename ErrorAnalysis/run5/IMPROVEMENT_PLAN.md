# run5 — Improvement Plan (baseline: 19.06% WER, full test set)

Based on `error_analysis/clusters.txt` and `error_analysis/confusions.txt` for this run. Ranked by expected impact.

## 1. Colloquial ↔ formal register flip-flopping (biggest lever)
Top substitutions go **both directions** on the same word pairs: `කියල⇄කියලා`, `කරන්නෙ⇄කරන්නේ`, `තියන⇄තියෙන`, `කථා⇄කතා`, `සමග⇄සමඟ`, `තුල⇄තුළ`. The model isn't mishearing — it's guessing which register (spoken vs. literary) the reference uses, because training data mixes both inconsistently across sources.
- **Fix:** normalize transcripts to one register convention (or tag data by register) across all four source corpora before the next fine-tune.

## 2. Word-boundary / conjunct inconsistency
Single-character deletions (`ය`, `ම`, `ද`, `ව` — 307 of top-25 deletions) and the ZWJ conjunct cluster (`්‍ය`, `්‍ර`, 652 samples) both trace back to inconsistent compounding/spacing and Unicode ZWJ encoding across source datasets.
- **Fix:** run an NFC/ZWJ-normalization pass + unify compound-word spacing convention over all transcripts. One-time, cheap data fix.

## 3. Religious/honorific register underrepresented
`වහන්සේ`-cluster (130 samples) still shows elevated WER, e.g. `මහාකස්සපයන්` → `මහා කශ්ෂපයන්`. Likely sermon/Dhamma-talk content is a small slice of training data.
- **Fix:** add more of that register if it matters for the product's domain.

## 4. Rare proper nouns / abbreviations
Low volume but high per-sample WER (`එස්.ඩබ්ලිව්.ආර්.ඩී .බණ්ඩාරනායක`, `ඩී.එස්.සේනානායක` mangled). Likely acoustic/OOV, not a data-labeling issue.
- **Fix:** lower priority; targeted augmentation only if abbreviated names matter for deployment.

## Recommended next steps
1. Normalize orthography (register + spacing + ZWJ) across all training transcripts — addresses #1 and #2 together, likely the largest chunk of the 47% wrong-sample rate.
2. Re-run fine-tuning with the same config as run5 (full fine-tune, lr 3e-5, bs64) on the normalized data.
3. Oversample honorific/religious-register audio if that domain matters.

Since run5 is a **full fine-tune** (not LoRA), the LoRA-specific failure modes in `../Analysis.md` (degenerate repetition, capacity limits) don't apply — this run's ceiling is data-quality-bound, not architecture-bound.
