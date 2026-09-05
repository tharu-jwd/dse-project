# E004: Teacher-Target English Behavior Replay

## Status

Complete. E004 preserves E003's Sinhala result while passing the frozen English
retention gate. It becomes the validated replay method for subsequent Sinhala
scaling experiments, but is not a release candidate because Sinhala error rates
remain far above 10%.

## Question

Can replaying untouched Whisper-small's behavior on the same 10% English audio
mixture preserve English recognition better than E003's raw LibriSpeech targets,
without surrendering E003's Sinhala gain?

## Controlled change

E004 holds E003's official base model, 10,000 Sinhala rows, 1,111 English audio
rows, replay ratio, wide rank-16 LoRA targets, seed, batch schedule, learning
rate, 500-step budget, validation set, and decoding fixed. The only intended
change is the target text for the English rows:

- E003: raw uppercase, unpunctuated LibriSpeech reference
- E004: untouched Whisper-small transcript from that same training audio

This is hard behavior replay, not test-set training or soft-logit distillation.
The teacher never sees LibriSpeech test-clean during target generation.

## Frozen teacher artifact

The private Kaggle teacher job used the same pinned Whisper-small weights and
English decoding settings as the retention benchmark. It generated 1,111
nonempty transcripts for 1,111 unique sample IDs and audio hashes in 156.14
seconds on a Tesla T4. Every teacher transcript differs byte-for-byte from its
raw LibriSpeech reference, primarily because Whisper emits natural casing and
punctuation; some outputs also preserve the base model's recognition errors,
which is intentional when rehearsing its existing behavior.

- source E003 manifest fingerprint:
  `71de2a4fd6ecec0418b0810fabf000d8e4dd40f4427dca0e1b0216b0b7485368`
- base-model weight SHA-256:
  `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`
- teacher-label artifact SHA-256:
  `5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6`
- E004 semantic training fingerprint:
  `d6e29790c9cd17f26055c98a92ce784edacb8599635d26ce3d177dc0cc816899`

## Decision rule

E004 uses the same Sinhala thresholds as E003: canonical WER at or below
101.69% and CER at or below 52.04%. Its English canonical WER may be at most
0.50 percentage points above untouched Whisper-small, and the upper end of the
paired 95% interval may be at most +1.00 point. It must also materially improve
English retention over E003. Report strict and canonical metrics together so a
formatting shift cannot be mistaken for better recognition.

## Training and Sinhala validation

The full private Kaggle run completed 500 steps on a Tesla T4. Training took
3,000.93 seconds and Sinhala validation generation took 62.69 seconds. Peak
allocated GPU memory was 1,802,875,904 bytes and mean training loss was 6.5580.
Every checkpoint archive from step 25 through step 500 was downloaded and its
SHA-256 independently matched the checkpoint index.

- final adapter SHA-256:
  `29f01b70a7a62abd6050c688bf1a09a8979429dd95eb0fe098237fce7803ae1c`
- Sinhala prediction SHA-256:
  `3fee2b28448f61e3b156f5fb04f455fd5df6a808dbbab46b8b1633f6fc5f43d3`
- checkpoint-index SHA-256:
  `21c272d3e5ca2e7813762afc8b49299373105a2ae7285f1b9f1ec0b831cba867`

| Sinhala validation metric | E002 | E003 | E004 |
|---|---:|---:|---:|
| Strict WER | 98.24% | 95.87% | 95.87% |
| Strict CER | 42.42% | 39.48% | 38.55% |
| Canonical WER | 97.53% | 95.07% | 95.48% |
| Canonical CER | 41.43% | 38.23% | 37.63% |

Against E003, canonical WER changed by +0.41 points (paired 95% interval -1.69
to +2.28) and CER by -0.60 points (-1.72 to +0.47). Both intervals include
zero, so E004 is statistically equivalent to E003 on this validation set. It
retains the small Sinhala gain over E002: canonical CER is 3.80 points lower
than E002 (interval -5.27 to -2.41), while the WER interval narrowly includes
zero. Teacher targets therefore did not meaningfully sacrifice Sinhala.

## English retention

The final adapter was evaluated on the unchanged 2,620-row LibriSpeech
test-clean benchmark. The first submission stopped during argument validation
because the frozen evaluator snapshot's display-label choices ended at E003;
no inference occurred. The second used the identical adapter inference path
under a supported internal label, then rewrote only output provenance fields to
E004. The prediction content was not transformed.

- benchmark SHA-256:
  `eb1d6f299f5fefde5b66fab450ffbc3b5bf2518ec9e64d3829c050579c6f2906`
- E004 prediction SHA-256:
  `8b4f65d8304bcf371ba190dba8023e544691b37af985bfa4f3803d34e3a4bc2d`
- inference time: 637.62 seconds on a Tesla T4

| Canonical English metric | Untouched | E002 | E003 | E004 |
|---|---:|---:|---:|---:|
| WER | 4.23% | 6.35% | 13.84% | 4.62% |
| CER | 1.92% | 2.89% | 5.88% | 2.02% |

Relative to untouched Whisper-small, E004 canonical WER increased 0.39
percentage points (paired 95% interval +0.17 to +0.58) and CER increased 0.09
points (-0.05 to +0.21). The point WER increase is below the frozen +0.50-point
limit and the interval's upper end is below +1.00, so the retention gate passes.
At word level 221 rows improved, 2,020 tied and 379 regressed; substitutions
rose by 226, deletions fell by 34, and insertions rose by 14.

E004 also conclusively reverses the prior forgetting: versus E002, canonical
WER falls 1.73 points (interval -2.04 to -1.46); versus E003 it falls 9.21
points (-9.65 to -8.81). Unlike E003, E004 strict output style remains close to
untouched Whisper, confirming that behavior targets preserve both formatting
and normalized recognition content.

## Conclusion and next experiment

E004 answers its question positively. Hard teacher-target replay on licensed
English training audio is sufficient to preserve the base model's standalone
English behavior at this scale while keeping the Sinhala gain. This validates
the replay mechanism, not the model as a whole: Sinhala canonical WER 95.48%
and CER 37.63% are still unusable.

Only 10,000 of v4's 182,665 training rows (roughly 12 of 220.88 training hours)
were used in E004. The next controlled axis is therefore Sinhala data scale,
not another transcript-style experiment or a longer pass over the same small
subset. E005 should expand to a deterministic 50,000-row Sinhala subset, keep
the E004 architecture and teacher-replay method, target roughly one effective
epoch, and preserve the same validation and English-retention gates. This
intermediate scale limits free-GPU risk before committing to all 220.88 hours.
