# Compute and Experiment Gate

Last verified 2026-09-06. The [GitHub Student Pack offer](https://education.github.com/pack)
currently states 5 Camber GPU hours, 40 CPU hours, and 50 GB storage per month.
[Camber's pricing page](https://www.cambercloud.com/pricing) states one credit is
USD 1 and lists its smallest on-demand GPU engine at 3 credits/hour. Product
entitlements and the exact GPU assigned can change, so the displayed engine,
VRAM, credit balance, and live price must be copied into the run config before
launch.

At that listed paid rate, USD 10 purchases at most 3.33 additional GPU hours;
with the five included hours the theoretical ceiling is about 8.33 hours. This
was never enough to promise several full 224-hour-corpus Whisper-small
experiments. It went unused through E007 (every experiment through E007 ran
on free Kaggle/Colab compute, not Camber). Starting with E009 (2026-09-06),
Camber is in actual use for bounded pilots that don't need Kaggle's larger
free allowance -- see [the Camber operating notes](../training/camber-cli.md)
for the verified CLI mechanics, environment gotchas (Python version mismatch
between the CPU and GPU images, a `numpy` version cap needed, no git checkout
in the job's workdir) and running total of GPU-hours actually spent so far.

## Current readiness and actual compute platform

Dataset v4 is frozen and fingerprinted, transcript review is locked, audio
quality has been audited, and local train/save/evaluate checks pass. E000-E002
ran on free Colab T4 sessions. Starting with E003, Colab was abandoned as the
primary platform after four consecutive `503 Service Unavailable` allocation
failures across three work cycles left no usable runtime; see the E003
experiment report for the failure record. Kaggle (free-tier T4/dual-T4
sessions, a separate weekly GPU-hour allowance from a different provider) has
been the sole execution platform since E003, and all of E003-E007 ran there.
Camber Cloud (GitHub Student Pack GPU credit) joined as a third platform
starting with E009 -- see [the Camber operating notes](../training/camber-cli.md)
for verified mechanics and the running GPU-hour total. No paid GPU run has
occurred on any platform.

The original Colab-specific isolation policy remains documented in
[the Colab operations policy](../training/colab-cli.md) for historical
reference and in case Colab is ever used again, but it does not describe
current operations. The actual, current operational policy for Kaggle is in
[the Kaggle operations policy](../training/kaggle-cli.md); for Camber, in
[the Camber operating notes](../training/camber-cli.md).

## Allocation

1. Run the untouched `openai/whisper-small` v4 validation baseline before
   spending any GPU credit. Use dataset v4 and do not access the locked test
   split.
2. Use a short bounded window for environment, mixed precision, data-loader,
   checkpoint upload/download, and deliberate-resume validation before any
   full run.
3. Use a short bounded pilot (500-1,000 steps) before a full run. Record
   samples/second, validation generation time, VRAM, checkpoint time, and
   billed time.
4. Recalculate full-run time from the slower of training and validation
   measurements. Reserve a 25% margin and enough headroom for recovery and
   artifact download.
5. Do not start a second recipe unless the measured plan leaves enough balance
   to finish it. Never use paid credits merely to debug data or code.
6. For runs projected close to a provider's per-session ceiling (Kaggle's is
   12 hours), split the run into checkpoint-resumed stages ahead of time
   rather than risking losing an entire session's progress. E007 uses this
   two-stage pattern for its full-dataset epoch.

Upload only the frozen data required by the provider. The local source
snapshots remain canonical. Download checkpoints, trainer state, predictions,
metrics, and run metadata before terminating the job. For E002 onward, do not
mount Google Drive; for Kaggle, do not rely on `/kaggle/working` surviving past
the session, and re-verify every downloaded hash locally before trusting it.

## Experiment order and English retention

Whisper was pretrained jointly on multilingual transcription and translation;
the [original Whisper paper](https://cdn.openai.com/papers/whisper.pdf) reports
benefits from joint multilingual/multitask training. Target-language-only
adaptation can nevertheless cause catastrophic forgetting, as demonstrated in
[Interspeech 2024 work on Whisper LoRA](https://www.isca-archive.org/interspeech_2024/xu24h_interspeech.html).
Recent controlled low-resource results show that parameter-efficient adaptation
can approach full fine-tuning with far fewer trainable parameters, but outcomes
are language-dependent ([AfricaNLP 2026](https://aclanthology.org/2026.africanlp-main.19/)).

Accordingly, run one factor at a time. The completed sequence through E007 is:

1. untouched Whisper-small on frozen validation, including Sinhala-only and
   code-switched slices (E000);
2. clean Sinhala wide-target LoRA (E001-E002) -- Sinhala improved but English
   retention regressed;
3. the same adapter recipe with a fixed, licensed English replay slice
   (E003, raw-reference targets -- failed retention) then teacher-target
   replay (E004 -- passed retention, established as the retention method);
4. a nested Sinhala data-scale curve holding that recipe fixed: 50,000 rows
   (E005), 100 hours (E006), then the complete 220.877-hour split (E007,
   **complete, both gates pass** -- canonical WER 81.71%, CER 26.15%,
   English retention statistically equivalent to E006) -- material Sinhala
   gains measured at every step, English retention passed at every step,
   Sinhala still far above the under-10% target;
5. two levers explored in parallel once E007 was under way rather than
   waiting on it: an automated LoRA rank/learning-rate search (E008,
   validated by a controlled follow-up, E010 -- rank=32, lr~2.3e-4 now the
   recommended default for future LoRA experiments on this recipe) and a
   Sinhala tokenizer vocabulary extension (E009 -- pilot run found unstable
   under this project's LoRA recipe for a confirmed, fixable reason; fix
   identified, re-run pending).

Measure English before and after every candidate on the same fixed English set.
Do not delete English words from Sinhala references or metrics: that would make
the reported Sinhala result easier without improving the recognizer. A separate
Sinhala-only slice answers the monolingual question honestly.

See [the plan's next-step priority order](plan.md#next-step-priority-order-after-e006e007)
for what comes after E007.
