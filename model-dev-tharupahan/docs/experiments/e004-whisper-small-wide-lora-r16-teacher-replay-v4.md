# E004: Teacher-Target English Behavior Replay

## Status

Teacher labels are frozen and training preparation is in progress. No E004
training result exists yet.

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
