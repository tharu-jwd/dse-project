# E005: Whisper-small wide LoRA, 50k Sinhala rows, teacher replay

## Status

Complete. E005 materially improves Sinhala over E004 and passes the frozen
English-retention gate. Sinhala error rates remain far above the under-10%
acceptance target.

## Question

Does increasing the deterministic v4 Sinhala-source subset from 10,000 to
50,000 rows for roughly one effective epoch materially reduce Sinhala WER and
CER while the E004 teacher-target replay method continues to preserve English?

This changes the Sinhala data scale and training exposure together. It does not
test a new transcript policy, LoRA architecture, learning rate, base checkpoint,
or English-preservation method.

## Frozen recipe

- Base: untouched `openai/whisper-small`
- Adaptation: wide LoRA rank 16 on `q_proj`, `k_proj`, `v_proj`, `out_proj`,
  `fc1`, and `fc2`
- Learning rate: `5e-5`
- Per-device batch: 4; gradient accumulation: 4
- Seed: `20260903`
- Sinhala source: deterministic speaker-balanced 50,000-row v4 train subset,
  60.31 hours, 471 speakers, including all 10,000 E004 source rows
- English replay: the 1,111 E004 LibriSpeech train-clean audio clips paired with
  frozen untouched-Whisper teacher transcripts
- Training mix: 50,000 Sinhala-source occurrences and 5,556 English replay
  occurrences (10.0007% English of the combined 55,556 rows)
- Training budget: 1,736 optimizer steps, approximately one pass on Kaggle's
  two T4 devices at effective batch 32
- Durable checkpoints: every 124 steps, including the final step 1,736
- Validation: the unchanged 206-row v4 validation bundle

## Replay limitation

Only 1,111 independently licensed English teacher-labelled clips are frozen.
To retain the E004 replay fraction, E005 uses five stable-key copies of every
clip plus one sixth occurrence of the first stable-key row. Every occurrence
gets a unique training ID while retaining its original audio hash. This tests
whether the already validated replay behavior survives Sinhala scaling, but it
does not test the benefit of greater English audio diversity. A later experiment
may expand unique English clips if E005 shows overfitting or retention failure.

## Immutable identities

- Sinhala 50k bundle SHA-256:
  `00fae26f174f6abb9f360ea9b91d064f17cf8ec609fe44d9375a2d94d8df104a`
- English source SHA-256:
  `0246f185b9f08a79e1eec91f7effb07e7912dd3199b00d78a72cfef61c54781b`
- Teacher-label SHA-256:
  `5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6`
- Validation bundle SHA-256:
  `c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2`
- E005 semantic training fingerprint:
  `e7146d640abf442b743cc5f65e8060bb1d11a40d73b8dc25932b1fe675f8bc32`

The local preflight reconstructed all 56 training shards from their real
artifacts and verified 50,000 Sinhala rows, 5,556 unique replay occurrences,
1,111 unique English audios, the validation hash, and the semantic fingerprint.

## Decision gates

Report strict and canonical WER/CER, error-operation counts, paired bootstrap
intervals, and row-level changes against E004. A Sinhala change is treated as
material only when the paired 95% interval excludes zero. Regardless of Sinhala
gain, English must pass the existing retention gate against untouched
Whisper-small: canonical WER point degradation no more than +0.50 percentage
points and the paired interval's upper end no more than +1.00 point.

The next experiment will be selected only after both evaluations. If Sinhala
improves materially and English passes, scale is promising; if English fails,
increase replay diversity or strength; if Sinhala is flat, additional raw scale
alone is not justified and the error breakdown must determine the next axis.

## Execution notes

Kernel version 1 crashed before loading any data: `run_e002_colab.py` is
intentionally copied into both `sinhala-asr-e003-inputs` and
`sinhala-asr-e003-runtime` (each dataset's own preparation script stages a
copy), so the runner's single-match file lookup found it twice and raised
before the first optimizer step.

A second, uncoordinated fix attempt (kernel version 2) removed
`sinhala-asr-e003-inputs` from the kernel's attached datasets entirely. That
resolved the duplicate match but removed the only dataset containing
`train-manifest.json` and the English-replay shards, so version 2 crashed
identically fast on the next lookup. Neither failure consumed meaningful GPU
time; both are preserved as attempt evidence.

The correct fix (kernel version 3) restored `sinhala-asr-e003-inputs` to
`dataset_sources` and anchored the runtime-directory lookup on
`whisper-small--model.safetensors` — a filename unique to
`sinhala-asr-e003-runtime` — instead of the duplicated `run_e002_colab.py`,
matching the pattern already used in `e004-english-evaluation.py`. All four
`one_file()` anchors were independently verified to resolve to exactly one
match against the real local dataset mirrors before the fix was pushed.

## Training and Sinhala validation

Kernel version 3 completed all 1,736 steps on Kaggle's two Tesla T4 devices as
planned: `trainer_state.json` records `train_batch_size: 8` (per-device 4 x 2
GPUs), giving the intended effective batch 32 once gradient accumulation (4)
is applied; `run-metadata.json`'s single `"gpu": "Tesla T4"` field only names
device 0 and is not evidence of a single-device run.
Training took 10,461.35 seconds (~2h54m) and Sinhala validation generation
took 61.01 seconds. Peak allocated GPU memory was 1,801,439,744 bytes. Mean
training loss was 2.9544 over the full 0.9999 effective epoch. All 14
checkpoint archives from step 124 through step 1,736 were downloaded and their
SHA-256 independently matched `checkpoint-index.json`; `status.json` reported
`{"state": "complete", "exit_code": 0}` and no traceback, CUDA-memory error, or
non-finite loss appeared anywhere in the run log.

- final adapter SHA-256:
  `cc4930df9d1207eee4e008db1e47f19dd8dc11763685696686fadeba3a5f42f3`
- Sinhala prediction SHA-256:
  `bffd39f12b89f5689cf4e311eab6e2a75b2a453d0bdce865ad8da916344e7248`
- checkpoint-index SHA-256:
  `6d17e98f429da5307972c2b2c6a879fa4a72541129044d65876fac98c56c49e0`

| Sinhala validation metric | E004 | E005 |
|---|---:|---:|
| Strict WER | 95.87% | 89.57% |
| Strict CER | 38.55% | 33.31% |
| Canonical WER | 95.48% | 88.39% |
| Canonical CER | 37.63% | 32.24% |

Against E004, canonical WER fell 7.09 percentage points (paired 95% interval
-9.99 to -4.00) and canonical CER fell 5.39 points (-6.99 to -3.80). Both
intervals exclude zero, so the Sinhala improvement from scaling 10,000 to
50,000 source rows is material, not sampling noise. Word
substitutions/deletions/insertions changed by -59/-13/+3; at word level, 77
rows improved, 96 tied, and 33 regressed; at character level, 125 improved, 40
tied, and 41 regressed. One of 206 validation rows is now an exact match — the
first exact row in this project's history of E000-E005. The adapter remains
far from usable, but the error pattern continues shifting from broad acoustic
collapse toward spelling and segmentation mistakes.

## English retention

The final E005 adapter was evaluated on the unchanged 2,620-row LibriSpeech
test-clean benchmark using a new `e005-english-evaluation` kernel (mirroring
`e004-english-evaluation.py`, reusing the same frozen runtime, benchmark, and
evaluator snapshot). The evaluation completed in 640.97 seconds on a Tesla T4.

- benchmark SHA-256:
  `eb1d6f299f5fefde5b66fab450ffbc3b5bf2518ec9e64d3829c050579c6f2906`
- E005 prediction SHA-256:
  `a8c8a1dd6e5acb718930344aaef2dbef2402a3cfa28e0cbb33c9d533adbbec42`

| Canonical English metric | Untouched | E004 | E005 |
|---|---:|---:|---:|
| WER | 4.23% | 4.62% | 4.52% |
| CER | 1.92% | 2.02% | 2.05% |

Relative to untouched Whisper-small, E005 canonical WER increased 0.29
percentage points (paired 95% interval +0.10 to +0.46) and CER increased 0.13
points (-0.01 to +0.24). The point increase is below the frozen +0.50-point
limit and the interval's upper end is below +1.00, so the retention gate
passes. Relative to E004, canonical WER changed -0.10 points (-0.25 to +0.05)
and CER changed +0.03 points (-0.05 to +0.12); both intervals include zero, so
E005 is statistically equivalent to E004 on English — scaling the Sinhala
training data fivefold did not measurably cost additional English retention
beyond what E004 already spent.

## Conclusion and next experiment

E005 answers its question positively: scaling the deterministic Sinhala
subset from 10,000 to 50,000 rows for one effective epoch, with the E004
teacher-replay method held fixed, produces a material Sinhala improvement
while the English-retention gate continues to pass with no measurable
additional cost over E004. Per the frozen decision rule, this means scale is
a promising axis and the next controlled experiment should continue it: build
a larger deterministic superset of the current 50,000-row selection (per the
project plan's nested data-scale curve) with the same architecture, teacher-replay
method, and step-per-epoch convention, and re-evaluate both gates before
deciding whether to advance toward the full 182,665-row/220.88-hour dataset.

## Decision rule outcome

Both frozen gates were checked against real, hash-verified evidence, not
aggregate point estimates alone: the Sinhala paired 95% interval excludes zero
in the improving direction (material), and the English canonical WER point
delta (+0.29pp) and interval upper bound (+0.46pp) both fall within the frozen
+0.50pp / +1.00pp limits (passes). Per the pre-registered rule, scale is
promising and E006 should continue scaling Sinhala data rather than changing
architecture, learning rate, or the replay method.
