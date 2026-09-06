# E006: Whisper-small wide LoRA, 100-hour Sinhala tier, teacher replay

## Status

Complete. E006 materially improves Sinhala over E005 and passes the frozen
English-retention gate. Scale remains useful at 100 hours, although the model
is still far above the under-10% acceptance target.

## Question

Does extending the nested Sinhala scale curve from E005's 50,000 rows and
60.31 hours to 100 hours materially reduce Sinhala WER and CER while the E004
teacher-target replay method continues to preserve English?

## Frozen recipe

- Base: untouched `openai/whisper-small`
- Adaptation: wide LoRA rank 16 on `q_proj`, `k_proj`, `v_proj`, `out_proj`,
  `fc1`, and `fc2`
- Learning rate: `5e-5`
- Per-device batch: 4; gradient accumulation: 4
- Seed: `20260903`
- Sinhala source: 82,835 deterministic v4 train rows totaling 100.0003 hours,
  built as a strict superset of all 50,000 E005 source rows
- English replay: 9,204 occurrences from the same 1,111 E004 teacher-labelled
  LibriSpeech clips, giving 10% of the combined 92,039-row mix
- Training budget: 2,870 optimizer steps, approximately one effective epoch
  on Kaggle's two T4 devices
- Durable checkpoints: every 205 steps, including final step 2,870
- Validation: unchanged 206-row v4 validation bundle

The limited English diversity remains a known constraint: all 1,111 unique
clips occur eight times, with a deterministic partial ninth copy of 316 clips.
The purpose is to hold the validated replay ratio constant while isolating the
effect of additional Sinhala data.

## Immutable identities

- Sinhala shard-set SHA-256:
  `2f55ff4ea8bb7e4eebfc4cd781e5358b907def6c9a5cebd9a54fad8c2a6246e7`
- English source SHA-256:
  `0246f185b9f08a79e1eec91f7effb07e7912dd3199b00d78a72cfef61c54781b`
- Teacher-label SHA-256:
  `5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6`
- Validation bundle SHA-256:
  `c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2`
- E006 semantic training fingerprint:
  `c156fc2be0dc7c579e3ad01406f0ed4ac9325ac597ca119d5c03d9d35c2c3c9b`

## Training and Sinhala validation

The private Kaggle run completed all 2,870 steps. Training took 18,710.22
seconds (5h11m50s), and validation generation took 73.08 seconds. Peak GPU
memory allocated on the recorded device was 1,801,491,968 bytes. Mean training
loss was 2.3180 over 0.9978 effective epoch. `status.json` records exit code
zero. All 14 checkpoint archives from step 205 through step 2,870 were
independently re-hashed and matched the durable checkpoint index.

- Final adapter SHA-256:
  `00a43073efe812c4aeae76f817b0e73f7ac56357f4e2175b6e8f05615ccbe711`
- Sinhala prediction SHA-256:
  `4e42acb9c9f82ff64bd21a038cb779737953c5cac3dadf60fcdec1ab6bb353fe`
- Checkpoint-index SHA-256:
  `eed69ae2b21899f9a7f11af95bd10c6b26be5650be82f39e13b18c4dc6476293`

| Sinhala validation metric | E005 | E006 |
|---|---:|---:|
| Strict WER | 89.57% | 85.33% |
| Strict CER | 33.31% | 29.82% |
| Canonical WER | 88.39% | 84.48% |
| Canonical CER | 32.24% | 28.74% |

Against E005, canonical WER fell 3.91 percentage points (paired 95% interval
-6.43 to -1.23) and canonical CER fell 3.50 points (-4.92 to -2.19). Both
intervals exclude zero, so the additional 39.69 hours produced a material
Sinhala improvement. Word substitutions and deletions fell by 27 and 29,
while insertions rose by 18; 55 rows improved, 119 tied, and 32 regressed at
word level. Character substitutions, deletions, and insertions fell by 62, 72,
and 40; 98 rows improved, 70 tied, and 38 regressed.

The improvement is smaller than E005's gain over E004, showing diminishing
returns from scale, and the model remains far above the under-10% target.
Nevertheless, the confidence interval demonstrates that the learning curve has
not yet plateaued at 100 hours.

## English retention

The final E006 adapter was evaluated on the unchanged 2,620-row LibriSpeech
test-clean benchmark. Inference took 649.45 seconds on a Tesla T4.

- Benchmark SHA-256:
  `eb1d6f299f5fefde5b66fab450ffbc3b5bf2518ec9e64d3829c050579c6f2906`
- E006 English prediction SHA-256:
  `9da80f2242457d282243451890ec859452994f44a1a6fee4841de2419463a66a`

| Canonical English metric | Untouched | E005 | E006 |
|---|---:|---:|---:|
| WER | 4.23% | 4.52% | 4.58% |
| CER | 1.92% | 2.05% | 2.06% |

Relative to untouched Whisper-small, E006 canonical WER increased 0.3483
percentage points (paired 95% interval +0.1457 to +0.5185) and CER increased
0.1363 points (+0.0013 to +0.2453). The WER point increase is below the frozen
+0.50-point limit and the interval upper bound remains below +1.00, so the
retention gate passes.

Relative to E005, canonical WER changed by only +0.0584 points (interval
-0.0456 to +0.1625), and CER by +0.0095 points (-0.0581 to +0.0692). Both
intervals include zero: E005 and E006 are statistically equivalent on English.
Increasing Sinhala exposure from 60.31 to 100 hours did not measurably worsen
retention under the teacher-replay recipe.

## Conclusion and next experiment

E006 passes both pre-registered gates. Sinhala WER and CER improve materially,
and English remains inside the retention limit. Therefore the next controlled
scale point is the complete 182,665-row, 220.88-hour v4 training split with the
same base model, adapter architecture, learning rate, one-effective-epoch
convention, and approximately 10% teacher replay.

At E006 throughput, a full-data epoch is projected near Kaggle's 12-hour
session ceiling. E007 must therefore be split at a durable checkpoint across
two private kernels rather than risking the whole epoch in one session. Phase B
must resume the exact Phase-A optimizer, scheduler, scaler, RNG, and trainer
state; it must not restart from the Phase-A adapter as a fresh optimization.
