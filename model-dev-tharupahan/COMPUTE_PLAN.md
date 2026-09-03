# Compute and Experiment Gate

Last verified 2026-09-03. The [GitHub Student Pack offer](https://education.github.com/pack)
currently states 5 Camber GPU hours, 40 CPU hours, and 50 GB storage per month.
[Camber's pricing page](https://www.cambercloud.com/pricing) states one credit is
USD 1 and lists its smallest on-demand GPU engine at 3 credits/hour. Product
entitlements and the exact GPU assigned can change, so the displayed engine,
VRAM, credit balance, and live price must be copied into the run config before
launch.

At that listed paid rate, USD 10 purchases at most 3.33 additional GPU hours;
with the five included hours the theoretical ceiling is about 8.33 hours. This
is not enough to promise several full 224-hour-corpus Whisper-small experiments.
It is enough to measure one capped smoke/pilot and possibly run one selected
recipe, depending on observed throughput. The code therefore refuses a run
whose `hourly_price_usd × estimated_hours × 1.25` exceeds its configured cap.

## Allocation

1. Spend no GPU time before the 2,000 gold candidates are reviewed and v2 is
   locked. Local train/save/resume/predict/evaluate smoke tests already pass.
2. Use at most 15 minutes of included time for environment, mixed precision,
   data-loader, checkpoint upload/download, and deliberate-resume validation.
3. Use at most 45 additional minutes for a 500–1,000-step Whisper-small pilot.
   Record samples/second, validation generation time, VRAM, checkpoint time,
   and billed time.
4. Recalculate full-run time from the slower of training and validation
   measurements. Reserve a 25% margin and at least 30 minutes for recovery and
   artifact download.
5. Do not start a second recipe unless the measured plan leaves enough balance
   to finish it. Never use paid credits merely to debug data or code.

Upload only the frozen v2 audio/manifest required by the provider. The local
source snapshots remain canonical. Download checkpoints, trainer state,
predictions, metrics, and run metadata before terminating the job.

## Experiment order and English retention

Whisper was pretrained jointly on multilingual transcription and translation;
the [original Whisper paper](https://cdn.openai.com/papers/whisper.pdf) reports
benefits from joint multilingual/multitask training. Target-language-only
adaptation can nevertheless cause catastrophic forgetting, as demonstrated in
[Interspeech 2024 work on Whisper LoRA](https://www.isca-archive.org/interspeech_2024/xu24h_interspeech.html).
Recent controlled low-resource results show that parameter-efficient adaptation
can approach full fine-tuning with far fewer trainable parameters, but outcomes
are language-dependent ([AfricaNLP 2026](https://aclanthology.org/2026.africanlp-main.19/)).

Accordingly, run one factor at a time:

1. untouched Whisper-small on frozen validation, including Sinhala-only and
   code-switched slices;
2. clean Sinhala full fine-tune;
3. the same winning recipe with a fixed, licensed English replay slice plus
   Sinhala–English code-switched data;
4. wide-target LoRA/DoRA under the same data and step budget.

Measure English before and after every candidate on the same fixed English set.
Do not delete English words from Sinhala references or metrics: that would make
the reported Sinhala result easier without improving the recognizer. A separate
Sinhala-only slice answers the monolingual question honestly.
