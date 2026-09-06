# E007 — Whisper-small wide LoRA, full v4 plus teacher replay

## Status

Prepared. Phase A has not yet started.

## Question

Does scaling the successful E006 recipe from 100 hours to the complete v4
training split continue to reduce Sinhala error while retaining the untouched
model's English capability?

## Frozen design

- Base model and adapter: Whisper-small, wide LoRA rank 16, unchanged from E006.
- Sinhala source: all 182,665 v4 training rows (220.877 hours), ordered by
  `sample_id`; selection fingerprint
  `78399cdea0f1b98a17aebe08db5fc8611eaae5be58f881fedf1c17fea3c10cbb`.
- English retention replay: 20,296 occurrences from the same 1,111 frozen
  LibriSpeech clips and untouched-model teacher targets used by E004–E006.
- Total: 202,961 training occurrences and 6,342 optimizer steps, approximately
  one effective epoch with the established global effective batch of 32.
- Checkpoints: every 453 steps. Phase A intentionally stops at step 4,077;
  Phase B must resume the exact saved optimizer, scheduler, scaler, RNG, and
  trainer state and continue to step 6,342.
- Evaluation remains the frozen 206-row Sinhala validation set followed by the
  frozen 2,620-row LibriSpeech English-retention benchmark.

The two-phase boundary keeps each Kaggle job below its 12-hour session ceiling.
It is one continuous training run, not a fresh adapter continuation. Phase B
will not be created until Phase A's exact checkpoint archive hash is known.

## Memory and failure controls

The 183 Sinhala audio shards are verified individually and converted one at a
time, avoiding simultaneous Arrow and Python copies of the full 12.7 GB corpus.
Only the 1,111 unique English audio rows are stored; replay occurrences reuse
the same immutable audio-byte objects in memory. Every durable checkpoint is
archived with hashes, and Phase B refuses to resume unless the step-4,077
archive hash and trainer state match the frozen handoff.

## Preparation verification

- Semantic source revision: `b5fe7a21855849508f6a8a9c0c29416031a82304`.
- Full Sinhala shard set: 183 shards, 12,689,696,104 bytes, combined SHA-256
  `ecd5bbffee8a644fd2db83385799838ca055cc832335aaa90de00ca7db35e6f9`.
- Local Kaggle staging preflight re-hashed every Sinhala shard and the frozen
  English/validation inputs. It produced 185 training shards containing
  183,776 stored rows; deterministic replay expansion yields the expected
  202,961 occurrences.
- The private Kaggle orchestration runtime was published with independently
  indexed copies of the generic runner and E007 orchestrator. The full audio
  input dataset must also be verified server-side before Phase A is submitted.

## Decision rule

Report strict and canonical Sinhala WER/CER with paired bootstrap deltas against
E006, then apply the unchanged English gate: WER point degradation no greater
than 0.50 percentage points and 95% CI upper bound no greater than 1.00 point.
No conclusion is recorded until both language evaluations finish.
