# Transcript Refiner Benchmark

## Purpose

Test whether Bedrock-accessible text models reproduce the transcript corrections
that a native Sinhala speaker verified against audio. Model outputs are isolated
suggestions and never modify dataset v3 or the pending v4 review.

## Benchmark set

The set contains the 295 disputed v4 review rows. Two rows marked `bad_audio`
are excluded, leaving 293 usable targets: 292 verified edits and one verified
unchanged original. Inputs contain only sample ID, language class, and the
normalized OpenSLR transcript; models receive no audio or previous v3 text.

This set is strongly selection-biased in favor of the earlier ChatGPT process:
the rows were selected precisely because that process had proposed a change.
It measures reproduction of those known corrections, not general false-positive
behavior. The remaining 100 unchanged controls must be completed before safely
estimating false rewrites on ordinary transcripts.

## Results

| Refiner | Proposed | Exact targets | Exact proposal precision | Improved rows | Worsened rows | Model→verified WER | Model→verified CER | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Earlier ChatGPT/v3 revision | 293 | 288 (98.3%) | 98.3% | 198 | 2 | 0.36% | 0.17% | unavailable |
| Claude Sonnet 4.6 | 85 | 65 (22.2%) | 75.3% | 68 | 5 | 25.25% | 2.51% | 35,378 |
| GPT-OSS 120B | 126 | 68 (23.2%) | 53.2% | 57 | 38 | 26.33% | 3.63% | 61,911 |
| DeepSeek V3.2 | 133 | 66 (22.5%) | 49.6% | 52 | 48 | 24.96% | 3.70% | 42,485 |
| Qwen3 Next 80B | failed | — | — | — | — | — | — | partial run |

The unchanged original has 30.59% WER and 3.45% CER relative to the verified
targets. Spaces do not contribute to character units, so many correct spacing
repairs tie rather than improve character distance.

Qwen was stopped after changing a sample ID. Guessing the intended alignment
would invalidate the benchmark and would be unsafe for batch application.

## Decision

No accessible model reproduces the earlier GPT-5.6 Sol web workflow. Among the
available Bedrock choices, Sonnet 4.6 is the preferred conservative suggestion
generator because it has the best exact-proposal precision, the lowest CER, and
far fewer harmful changes. It must still be used as a proposal source, not an
automatic label authority.

Do not use GPT-OSS 120B or DeepSeek V3.2 for unattended corrections under the
current prompt. Their higher proposal counts do not translate to higher exact
coverage and they introduce substantially more harmful rewrites.

After the 100 controls are audio-reviewed, rerun this benchmark including those
negative examples. Then estimate real correction precision, missed-error rate,
and the manual review burden before processing any training-label batch.

## Reproduction

```bash
PYTHONPATH=src python scripts/export_verified_refinement_benchmark.py \
  --queue reports/review/v4-evaluation-queue.parquet \
  --adjudications reports/review/v4-evaluation-adjudications.jsonl \
  --input-output reports/label-refinement/verified-293-input.json \
  --truth-output reports/label-refinement/verified-293-truth.json

PYTHONPATH=src python scripts/run_bedrock_refinement_batches.py \
  --input reports/label-refinement/verified-293-input.json \
  --output-dir reports/label-refinement/verified-293-sonnet \
  --model global.anthropic.claude-sonnet-4-6 --batch-size 25

PYTHONPATH=src python scripts/score_refinement_benchmark.py \
  --truth reports/label-refinement/verified-293-truth.json \
  --predictions reports/label-refinement/verified-293-sonnet/combined.json \
  --output reports/label-refinement/verified-293-sonnet/score.json
```
