# E016 — Honest evaluation of Yohan run7

## Status

Preparing a pinned comparison-only evaluation. No inference result exists yet.

## Question

How does Yohan's latest full Whisper-small checkpoint perform on this
project's transcript-disjoint v6 validation and frozen English-retention
benchmark when evaluated with this project's fixed decoding and metrics?

## Frozen design

- Model: `Yohan2003/whisper-small-sinhala`, subfolder
  `models/run7-v4-lr3e-5-bs64`, pinned to repository commit
  `bdaa42b21afa3923e84e6a98d54cfc984c4c99c3`.
- Sinhala evaluation: all 6,493 v6 validation rows, Sinhala transcription
  prompt, beam width 5, and no-repeat 3-gram constraint.
- English evaluation: all 2,620 frozen LibriSpeech test-clean rows, English
  transcription prompt, greedy decoding, and no-repeat 3-gram constraint.
- Preserve row-level predictions and recompute strict/canonical WER, CER,
  operation counts, and subgroup results locally.
- External test remains locked. Yohan's checkpoint is comparison-only and is
  not authorized as a training starting point by this experiment.

## Why Yohan's reported 19.06% is insufficient

His `stratified_v4` split cannot be verified as speaker-disjoint from the
available split code, and his WER normalization is more permissive than ours.
This experiment therefore does not reuse his reported score as evidence of
performance on our evaluation protocol.
