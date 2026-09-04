# E002: Larger Wide-LoRA Pilot

## Status

Ready to run; E002 training has not started. The untouched/E001
English-retention gate and deliberate checkpoint-resume smoke both passed.

## Question

Does extending the validated E001 recipe from 100 to 500 steps on a larger,
nested v4 training subset continue to reduce Sinhala errors without materially
damaging standalone English recognition?

## Frozen configuration

- Base: official `openai/whisper-small`, frozen
- Adapter: wide LoRA, rank 16, alpha 32, dropout 0.05
- Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
- Training subset: deterministic 10,000 v4 train rows, including 375
  Latin-only rows; this strictly contains the E001 selection
- Optimization: 500 steps, batch 4, accumulation 4, learning rate `5e-5`
- Checkpoint/log interval: 25/5 steps
- Sinhala evaluation: frozen 206-row v4 validation split
- English evaluation: full 2,620-row LibriSpeech test-clean split
- Test split: locked and unavailable to this run
- Compute: free Colab T4 only; no Google Drive mount

The official source file SHA-256 is
`7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`.
Benchmark identity uses a semantic content fingerprint over IDs, references,
and audio hashes. A generated Parquet file hash is also recorded for transport
integrity, but it may differ across PyArrow versions without content drift.

## Gates before training

1. [x] Evaluate untouched Whisper-small and E001 on the frozen English benchmark.
2. [x] Record strict/canonical English WER and paired error-count changes.
3. [x] Complete a two-step remote smoke and deliberate checkpoint resume.
4. [x] Verify all input hashes and confirm the estimated run remains inside the
   free-compute budget.

## Prerequisite English baseline

Untouched Whisper-small completed the full 2,620-row benchmark on a free Colab
T4 in 379.91 seconds:

| Metric | Result |
|---|---:|
| Strict WER | 98.83% |
| Strict CER | 100.38% |
| Canonical WER | 4.23% |
| Canonical CER | 1.92% |

The high strict errors are expected protocol evidence rather than an English
recognition failure: LibriSpeech references are uppercase and unpunctuated,
whereas Whisper emits ordinary casing and punctuation. Canonical scoring
removes those presentation differences and is the primary English-retention
metric. Strict results remain recorded so normalization cannot conceal a
content change.

The first CLI upload of the 346 MB prepared Parquet failed with a broken pipe.
The T4 runtime remained healthy, so the job instead downloaded the immutable
official source, verified its published SHA-256 and all rows, and continued.
This transport failure and recovery are preserved in attempt
`english-baseline-001`; no dataset or inference setting changed.

Prediction SHA-256:
`96d029665c2449c410bce6bace47476b0f400ede41cbf1b89070edd4c789d9af`.
Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1, and PEFT 0.20.0
were recorded with the run.

The original E001 adapter produced 4.2978% canonical WER and 1.9288%
canonical CER. Relative to untouched Whisper-small, changes were +0.0640 and
+0.0048 percentage points respectively. Paired 95% intervals for both changes
included zero, so there is no demonstrated English regression after E001. This
passes the English-retention prerequisite for the E002 smoke and training run.

## Checkpoint-resume smoke

The first diagnostic was rejected because phase A used a one-step scheduler and
phase B a two-step scheduler. Although state loaded, the schedules were not
identical; its artifacts were retained under rejected-schedule-mismatch attempt
names and are not treated as passing evidence.

The corrected diagnostic used a two-step schedule in both phases and deliberately
stopped phase A after step 1. Step 1 reported loss 12.55, finite gradient norm
21.81, and learning rate `5e-5`. The complete checkpoint was downloaded and
verified locally, then re-uploaded and loaded into a fresh training process.
Step 2 reported loss 12.43, finite gradient norm 20.61, and the expected learning
rate `2.5e-5`. Checkpoint archive hashes were:

- step 1: `32a08667f121a8cbce401f11e878d46f9b242002e060dda7378d33066a520137`
- step 2: `46be87c79c2e7c716fc1d8f253c8519888bf083f14740d832f31f56aef4cecc4`

All ten training-shard hashes and the validation-bundle hash passed remote
preflight. Observed compute plus checkpoint packaging projects the 500-step run
at roughly 40 minutes, leaving substantial margin within an ordinary free-T4
session. The run is therefore cleared to start without paid compute.

The experiment follows the failure containment and locally verified checkpoint
procedure in [the Colab operations policy](../training/colab-cli.md). A failed
attempt is retained separately and never overwritten.

## Decision rule

Proceed from E002 only if Sinhala validation improves beyond E001 without a
practically unacceptable English regression. If English regresses, the next
controlled experiment adds licensed English/code-switched replay while holding
the adapter, learning rate, Sinhala subset, and step budget constant.
