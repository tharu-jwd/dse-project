# E012 — Full-parameter fine-tune: learning-rate x replay-ratio search

## Status

**Running (version 2), after a diagnosed and fixed crash in version 1.**
Kernel `tharupahan/sinhala-asr-e012-full-finetune-lr-replay-search`
submitted to Kaggle (T4, `enable_internet: false`) on 2026-09-07. This is the
search step of [plan.md item 7](../project/plan.md#7-compare-full-whisper-adaptation-with-lora)
-- de-risking the full-fine-tune LR and replay ratio cheaply before committing
to the ~15-19h full run, per direct owner decision.

**Version 1 crashed at trial 5 of 9** (`replay20-lr5e-05`) with
`RuntimeError: basic_ios::clear: iostream error` inside
`torch.serialization.save` -- a disk-full error, not a training failure.
Root cause, confirmed directly from the trial's own log (not guessed):
full-parameter checkpoints are ~3GB each (`model.safetensors` ~967MB +
AdamW `optimizer.pt` ~1.9GB fp32 states), unlike LoRA's few-MB adapters, and
the kernel never deleted a completed trial's checkpoint before starting the
next -- by trial 5, 4 prior trials' checkpoints (~12GB) plus the crashing
trial's own in-progress write exceeded Kaggle's working-disk quota. Fixed by
deleting each trial's `checkpoint-*`/`final` directories immediately after
reading its `trainer_state.json`, before the next trial starts -- at most
one trial's ~3GB is ever resident on disk. Training and evaluation
themselves were never the problem: trial 5's own log shows it completed
100 steps and eval successfully (`eval_wer=0.9225`, `eval_cer=0.2901`)
before crashing on the save.

The 6 trials that did complete before the crash (all of replay10's 3 LRs,
2 of replay20's 3 LRs) already show a clear, consistent pattern worth
recording even though the search must be rerun in full: at this 100-step
pilot scale, only the highest learning rate tested (5e-5) shows real
convergence -- the two lower rates (1e-6, 5e-6) barely move off the
untouched model's initialization (`eval_wer` 1.01-1.13, i.e. at or above
100%) while 5e-5 reaches `eval_wer` ~0.91-0.92 and, more tellingly,
`eval_cer` ~0.29 on both replay ratios tested. Consistent with expectations
for full-parameter updates needing a meaningfully larger step than LoRA's
adapter subspace; not yet enough trials to call a replay-ratio winner.

## Question

Neither LoRA's winning learning rate (2.345e-4, from E008/E010) nor its 10%
English-replay ratio (fixed once in E004, never itself searched) is expected
to transfer to full-parameter fine-tuning, which moves every weight instead
of a small adapter subspace. Which combination of learning rate (log-spaced,
1e-6 to 5e-5) and replay ratio (10%/20%/30%) gives the best cheap-pilot
Sinhala validation result without an early sign of English forgetting?

## Frozen design

- Base model: stock `openai/whisper-small`, `method: "full"` (all parameters
  trainable, no LoRA) -- confirmed working via a local 2-step smoke test
  (`configs/training/diagnostics/tiny-cpu-full-finetune-smoke.json`) before
  spending any Kaggle time.
- Grid: 3 learning rates (1e-6, 5e-6, 5e-5) x 3 replay ratios (10%, 20%, 30%)
  = 9 trials, 100 steps each, batch 4 x accumulation 4 (1,600 occurrences per
  trial) -- same per-trial pilot scale as E008's LoRA rank/LR search.
- Data: v5's `openslr-train-v2` (see
  [the dataset spec](../data/dataset.md#dataset-v5-openslr-only-trainvalidation-external-only-test-2026-09-07))
  for Sinhala, the real E004 teacher-replay pool (1,111 LibriSpeech clips,
  untouched-model `teacher_text` targets, hash-verified against the same
  `TEACHER_SHA256` E004-E007 used) for English replay, and a 200-row
  deterministic slice of v5's `openslr-validation-v2` for eval. All three
  repackaged as self-contained (embedded-audio) parquets so the Kaggle
  dataset transport doesn't depend on external file paths:
  `scripts/data/prepare_teacher_replay_manifest.py`,
  `scripts/data/package_full_finetune_search_kaggle_inputs.py`,
  `scripts/data/build_full_finetune_search_manifests.py`.
- Objective, deliberately different from E008's WER-only search: log
  `eval_wer`, `eval_cer`, `eval_loss` (Sinhala validation) and the final
  training loss on the replay-inclusive batch stream for every trial, as a
  cheap early forgetting signal. The real English-retention check (frozen
  2,620-row LibriSpeech benchmark) is not run on all 9 trials -- it belongs
  on the narrowed winner only, per plan.md item 7's own search plan.
- Kaggle job structure: `kaggle/e012-full-finetune-search/`, reusing
  `sinhala-asr-e003-runtime` (offline wheels + staged whisper-small) as in
  every prior Kaggle job, plus two new datasets built this session:
  `sinhala-asr-e012-search-inputs` (the manifests + self-contained audio,
  ~344MB) and `sinhala-asr-e012-orchestration-runtime` (`src/sinhala_asr` +
  `scripts/training/train.py`, PYTHONPATH-imported, never pip-installed).

## Result

Pending -- kernel still running as of this write-up. Will be filled in once
`e012-search-results.json` is downloaded and verified.

## Next step

1. Read off the grid's best Sinhala-validation config.
2. Run the real English-retention check (frozen LibriSpeech benchmark) on
   that one config only.
3. If it passes, validate with one longer controlled run (E010-style)
   before committing to the full ~15-19h run.
