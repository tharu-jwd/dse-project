# Untouched Whisper-small v4 Baseline

## Run identity

- Model: `openai/whisper-small` with unchanged weights
- Dataset fingerprint: `d232747fbf019f06a6449404d3d0251e8f4547ed02471482c07d85014c81abdb`
- Split: 206 audio-verified validation rows; test remained locked
- GPU/runtime: Tesla T4 on free Colab, batch 8, 73.09 seconds
- Decoding: Sinhala transcription, 64-token cap, `no_repeat_ngram_size=3`
- Prediction SHA-256: `ed9e1d548335e33451cb800107dadca1aa37a8f33053dcc624623f5b0821452e`

All IDs, references, and encoded-audio hashes matched frozen v4 before scoring.

## Results

| Metric | Strict | Canonical |
|---|---:|---:|
| WER | 141.74% | 141.21% |
| CER | 92.52% | 92.44% |

Strict word operations were 904 substitutions, 61 deletions, and 407
insertions over 968 reference words. Twelve rows were automatically marked as
possible repetition or hallucination. Sinhala-only WER/CER was 141.51%/92.24%
on 204 rows. Two Latin-only rows cannot measure general English retention.

All three validation speakers performed poorly (135.06%–150.35% WER).
Predictions commonly collapsed into malformed, repetitive Sinhala-like
fragments, indicating broad zero-shot recognition failure rather than a
localized speaker, reference, or punctuation problem.

## Decoding diagnostic

An earlier run used a 225-token cap without a repetition constraint. It
produced 100 repetition/hallucination flags and 329.03% strict WER. That run is
retained as a decoding diagnostic, not the official accuracy baseline. The
guard reduced those flags to 12, so decoding explained much of the first
number, but the guarded result still demonstrates severe zero-shot weakness.

The next controlled step is a short wide-target LoRA/DoRA pilot.
Full-parameter fine-tuning is out of scope. English retention must be measured
on a separate fixed English set before and after adaptation.
