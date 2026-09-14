# E015 — Full-parameter Whisper-small, one v5 epoch with v6 validation

## Status

Preparing immutable inputs. The local checkpoint/resume regression passed;
the exact packaged Kaggle runtime smoke and measured training have not started.

## Question

Does the E012/E014 winning recipe scale from a repeatedly sampled pilot to the
complete leakage-controlled training corpus, while canonical English WER stays
at or below the owner-approved 10.00% ceiling?

## Frozen design

- Start checkpoint: untouched `openai/whisper-small`, not an E014 final model.
- Training: all 165,055 v5 OpenSLR rows plus 18,339 deterministic occurrences
  of 1,111 teacher-behavior English clips (10.00% of the combined mixture).
- Validation monitor: 200 deterministic v6 transcript-disjoint rows, covering
  all 52 held-out speakers (192 Sinhala-only, 8 Latin-only).
- Final validation: all 6,493 v6 transcript-disjoint rows.
- External test: locked and absent from training kernels.
- Full-parameter adaptation, 5e-5, linear schedule, 573 warm-up steps,
  effective batch 32, seed 20260903.
- One-epoch ceiling: 183,394 training occurrences and approximately 5,732
  optimizer steps.
- Phase A deliberately stops and checkpoints at step 3,300. Phase B resumes
  optimizer, scheduler, scaler, RNG, and trainer state and stops at step 5,732.

## Preflight requirements

1. Verify every input asset and reused E007 audio shard by SHA-256.
2. Filter reused E007 audio by the immutable v5 allow-list and verify sample,
   speaker, transcript, and audio-hash identity for all 165,055 rows.
3. Apply the English decoder prefix only to teacher-replay rows; default Sinhala
   source and validation rows to the Sinhala transcription prefix.
4. Preserve the phase-boundary checkpoint with a per-file hash inventory.
5. Preserve row-level monitor predictions outside any cleanup path.
6. Prove resume on the exact E015 entry point before the measured launch.

## Preflight evidence

- The generic trainer's mixed-language collator regression passes: a single
  batch receives distinct Sinhala and English decoder prefixes and restores the
  Sinhala inference default afterward.
- A local CPU full-parameter Whisper-tiny run stopped at step 1, produced a
  complete trainer checkpoint, resumed from it, and reached step 2. This proves
  the generic phase-boundary mechanism; the packaged Kaggle smoke remains the
  final launch gate because it also checks transported assets and GPU runtime.

## Known limitation corrected from E012/E014

The generic trainer used in E012/E014 initialized a Sinhala tokenizer prefix
globally and did not switch it per English replay row. The audio and English
benchmark remained disjoint, so this was not leakage, but the replay treatment
was not language-prompted as intended. E015 uses explicit per-row
`decoder_language`: `en` for teacher replay and `si` otherwise. This correction
means E015 is the first clean full-data test of the intended replay recipe and
must not be described as changing only data scale from E014.

## Decision rule

After the first epoch, select a checkpoint using validation only. Continue no
further unless Sinhala improves materially, canonical English WER is at most
10.00%, monitor/full-validation behavior shows no collapse, artifacts are
complete, and remaining free GPU allocation covers a resumable continuation.
