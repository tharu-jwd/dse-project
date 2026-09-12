# E014 — Full-finetune learning-rate and English-retention validation

## Status

Running as Kaggle kernel version 2 under the personal account
(`tharupahan/sinhala-asr-e014-validation`) from 2026-09-12.

Version 1 failed before training after 55 seconds. The kernel supplied the new
`lr_scheduler_type` and `warmup_steps` fields, but its reused E012
orchestration dataset contained an older `TrainConfig` and rejected both as
unknown. No optimizer step ran and GPU cost was negligible. The failure log is
stored under
`reports/experiments/e014-full-finetune-validation/attempts/kaggle-personal-v1/`.

The demonstrated failure was fixed by publishing
`tharupahan/sinhala-asr-e014-orchestration-runtime` from the current committed
training entry point and package, then repointing the kernel to that immutable
runtime. Version 2 was submitted only after Kaggle indexed and exposed every
runtime file.

## Question

Which bounded full-parameter Whisper-small configuration learns Sinhala while
remaining safe enough on English to justify an expensive multi-epoch run?

## Arms

Both arms use the same deterministic Sinhala/replay manifests, 10% frozen
teacher-behavior English replay, 500 optimizer steps, and the same 200-row
Sinhala validation slice:

1. Learning rate 5e-5 with linear scheduling.
2. Learning rate 1e-5 with cosine scheduling.

After each arm, score the resulting full checkpoint on the frozen 2,620-row
LibriSpeech test-clean English-retention benchmark. The full fine-tune remains
unauthorized until these results are independently verified.

## Required result handling

- Verify CUDA hardware and all artifact hashes from runtime metadata.
- Independently rescore Sinhala and English per-row predictions locally.
- Compare both arms against each other and the untouched English baseline.
- Apply the existing English-retention gate.
- Explain metric changes and select or reject a full-run configuration.
- Preserve any failure and do not retry without a demonstrated fix.
