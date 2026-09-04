# Wide-LoRA Pilot Results

## Configuration

- Base model: `openai/whisper-small`, frozen
- Adapter: rank 16, alpha 32, dropout 0.05
- Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
- Trainable parameters: 6,488,064 of 248,222,976 (2.61%)
- Training data: deterministic v4 pilot with 2,000 rows, 2.42 hours, all
  471 training speakers, 1,925 Sinhala-only and 75 Latin-only rows
- Optimization: 100 steps, batch 4, accumulation 4, effective batch 16,
  learning rate `5e-5`
- Hardware: free-Colab Tesla T4
- Training/evaluation time: 407.18/68.44 seconds
- Peak allocated GPU memory: 1,311,157,760 bytes
- Prediction SHA-256:
  `5311e5001b335e3f62a0205a3f10230e0100f562b9700095876ba45bac92e09f`

The adapter and checkpoints are stored under
`MyDrive/sinhala-asr/wide-lora-100-v1`. Validation used the same 206 v4 rows and
guarded decoding as the untouched baseline. The test split remained locked.

## Paired result

| Metric | Untouched | LoRA 100 | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Strict WER | 141.74% | 114.26% | -27.48 pp | -19.39% |
| Strict CER | 92.52% | 83.79% | -8.73 pp | -9.44% |
| Canonical WER | 141.21% | 114.18% | -27.03 pp | -19.14% |
| Canonical CER | 92.44% | 83.86% | -8.58 pp | -9.28% |

The paired bootstrap 95% interval for strict WER change is -34.61 to -20.29
percentage points, so the measured improvement is not sampling noise on this
validation set. Per row, 122 improved, 44 tied, and 40 regressed. All three
speakers improved by 16.38–38.46 points.

## What changed

Word substitutions fell from 904 to 785 and insertions from 407 to 138, while
deletions rose from 61 to 183. Repetition/hallucination flags fell from 12 to
one. Thus most of the early WER gain came from suppressing excessive malformed
output, partly by emitting less—not yet from accurate transcription. Neither
model produced an exact transcript among the 206 rows, and LoRA predictions
remain unusable.

This run validates the adapter implementation, memory fit, throughput, and a
positive learning signal. It does not validate English retention: v4 validation
contains only two Latin-only clips. The next experiment should use a larger
nested training subset and must add a fixed standalone-English evaluation set.
