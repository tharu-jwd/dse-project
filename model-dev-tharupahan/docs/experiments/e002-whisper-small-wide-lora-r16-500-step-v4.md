# E002: Larger Wide-LoRA Pilot

## Status

Prepared; E002 training has not started. The prerequisite untouched-English
baseline is complete; the E001-English comparison is waiting for the original
E001 adapter artifact.

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

1. Evaluate untouched Whisper-small and E001 on the frozen English benchmark.
2. Record strict/canonical English WER and paired error-count changes.
3. Complete a two-step remote smoke and deliberate checkpoint resume.
4. Verify all input hashes and confirm the estimated run remains inside the
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

The experiment follows the failure containment and locally verified checkpoint
procedure in [the Colab operations policy](../training/colab-cli.md). A failed
attempt is retained separately and never overwritten.

## Decision rule

Proceed from E002 only if Sinhala validation improves beyond E001 without a
practically unacceptable English regression. If English regresses, the next
controlled experiment adds licensed English/code-switched replay while holding
the adapter, learning rate, Sinhala subset, and step budget constant.
