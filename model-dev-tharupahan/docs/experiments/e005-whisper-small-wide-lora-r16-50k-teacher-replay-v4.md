# E005: Whisper-small wide LoRA, 50k Sinhala rows, teacher replay

## Status

Prepared and locally preflighted. The private Kaggle input upload is in
progress; no E005 training or evaluation result exists yet.

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
