# E014 — Full-finetune learning-rate and English-retention validation

## Status

Complete. Kaggle kernel version 2 under the personal account
(`tharupahan/sinhala-asr-e014-validation`) finished on 2026-09-13. Neither
arm is eligible for more repeated passes over the small pilot manifest: both
overfit it. Under the original pre-registered English gate, the high-rate arm
also failed; under the owner-revised absolute 10% ceiling adopted on
2026-09-14, both arms pass English acceptance.

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

The immutable untouched comparison is the existing per-row prediction file
with SHA-256
`96d029665c2449c410bce6bace47476b0f400ede41cbf1b89070edd4c789d9af`.
Its canonical baseline is 4.2338% WER and 1.9240% CER. The pre-registered
retention gate is unchanged: canonical WER point degradation no greater than
+0.50 percentage points and the paired 95% interval's upper bound no greater
than +1.00 point.

## Required result handling

- Verify CUDA hardware and all artifact hashes from runtime metadata.
- Independently rescore Sinhala and English per-row predictions locally.
- Compare both arms against each other and the untouched English baseline.
- Apply the existing English-retention gate.
- Explain metric changes and select or reject a full-run configuration.
- Preserve any failure and do not retry without a demonstrated fix.

## Results

| Arm | Effective epochs | Train time | Final train loss | Sinhala WER | Sinhala CER | English WER | English CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-5, linear | 10 | 1.484 h | 1.771 aggregate; 0.00695 last window | 98.81% | 34.61% | 5.7794% | 2.5486% |
| 1e-5, cosine | 10 | 1.488 h | 2.978 aggregate; 0.07974 last window | 114.30% | 52.08% | 4.7044% | 2.1482% |

The Sinhala figures are the trainer's final generation metrics on the frozen
200-row slice. The English figures were independently recomputed locally from
the saved 2,620-row prediction files using the canonical evaluator. The full
kernel took about 3.19 hours including both Sinhala evaluations and both
English inference passes; compute cost was USD 0 on Kaggle's free allocation.

The run used CUDA with PyTorch 2.10.0+cu128. Trainer state reports a global
per-step batch of 8 and 10 passes over the 1,600-row pilot manifest by step
500. Both arms therefore answer a *small-data repeated-exposure* question,
not what one epoch over the full 199.49-hour v5 training split will do.

## Interpretation

E014 invalidates the proposed 500-step bridge to a multi-epoch run. E012's
5e-5 arm measured 90.58% WER / 29.08% CER at step 100 (epoch 2), whereas the
same 5e-5/replay-10 recipe reaches 98.81% / 34.61% at step 500 (epoch 10).
Training loss meanwhile falls almost to zero. This is the signature of severe
memorization of the 1,600-row search manifest rather than continued
generalization. The 1e-5/cosine arm retains English better but generalizes
even worse to Sinhala; lower validation loss does not translate into better
generated sequences here.

This experiment changes learning rate and schedule together, so it supports a
recipe-level comparison only. It cannot attribute the difference separately
to learning rate or scheduler.

The kernel accidentally removed each arm's Sinhala prediction parquet when it
deleted the `final/` directory to control output size. Consequently the
trainer metrics are recoverable, but independent Sinhala rescoring, paired
row-level comparison, and qualitative error inspection are not. This is an
observability defect in the experiment harness and must be fixed before the
next run. English predictions, trainer state, resolved configuration, runtime
metadata, and the complete log were preserved.

### Original English gate and current reclassification

The 5e-5/linear arm fails decisively: canonical WER degrades by 1.546
percentage points, with a paired 95% interval of +1.324 to +1.770 points.
The 1e-5/cosine arm passes the pre-registered gate, narrowly on the point
threshold but comfortably on its uncertainty threshold: +0.471 points, 95%
interval +0.251 to +0.679 points. Its regression is nevertheless real (389
rows regressed versus 205 improved), driven mainly by 215 additional word
substitutions and 37 insertions. English preservation alone could not rescue
that arm under the original decision because its Sinhala result was the worse
of the two. Under the current absolute 10.00% policy, both arms pass; the
deltas and intervals remain important diagnostics rather than rejection
criteria.

## Decision and next experiment

Do **not** continue either final E014 checkpoint or authorize an unchecked
multi-epoch run. On 2026-09-14 the owner changed English acceptance to an
absolute canonical WER ceiling of 10.00%. Both E014 arms now pass that current
bar, including the stronger Sinhala arm at 5.7794%; the planned 100-step
English rerun would therefore no longer answer a blocking question.

Proceed instead to a fresh, checkpointed full-v5 run from Whisper-small using
5e-5, linear scheduling, and 10% teacher replay. Cap the first stage at one
full-data epoch, preserve Sinhala predictions outside checkpoint cleanup, and
measure intermediate checkpoints so training can stop before validation turns.
Run the full frozen English benchmark on the selected checkpoint before any
second epoch. E014 changed two factors together, so this advances a measured
recipe—not a claim that learning rate alone caused the result.

## Artifact trace

Downloaded outputs are under
`reports/experiments/e014-full-finetune-validation/attempts/kaggle-personal-v2/`.
Key SHA-256 values:

- kernel log: `e01cb4d644616942d6845409d756b7fbe50c95d0104982173d91ae339bfbcc37`
- high-rate English predictions: `b3d1d998d67e62bc435db98cb52796a4bee24ad55434c39d932af76217c8473c`
- conservative English predictions: `6a460efefb3401512571ca5e3cb0bb74299c95f6cba24f313c2a3bd0b89f7028`
- combined runtime result: `52e64cac70e9de8db3b9ff3853a480018e12e6e3e699c3611c99c36c190087fa`
