# External Review: whisper-based-sinhala-asr

Evidence-graded review of `/Users/tharupahan/Code/ASR-prior-works/whisper-based-sinhala-asr`
(GitHub `isurulucky/whisper-based-sinhala-asr`), a small, independent, tutorial-
style repository -- a third, separate Sinhala Whisper fine-tuning effort from
the ones already reviewed in [the ASR-Finetune external
review](asr-finetune-external-review.md). Same evidence labels: **Verified
artifact**, **Recorded result**, **Interpretation**, **Unknown**.

## What it is

A minimal, single-notebook reference implementation, explicitly derived from
the canonical [HuggingFace Whisper fine-tuning
blog post](https://huggingface.co/blog/fine-tune-whisper) (stated in its own
README). Not a research project -- a worked example. *(Verified artifact.)*

## Data and technique

- Source: a single OpenSLR-52 shard (`asr_sinhala_0.zip`), not the full
  corpus -- a small subset. *(Verified artifact.)*
- **Full-parameter fine-tuning**, not LoRA: plain
  `WhisperForConditionalGeneration.from_pretrained(...)` followed directly by
  `trainer.train()`, no adapter library involved anywhere in the notebook.
  *(Verified artifact.)*
- `openai/whisper-small`, LR `1e-5`, batch 16, 1000 steps, `predict_with_generate`,
  `metric_for_best_model="wer"`. *(Verified artifact.)*
- Evaluation: raw `jiwer`-backed WER via `evaluate.load("wer")`, computed on
  whatever the notebook's decoded strings happen to be -- **no Sinhala-specific
  normalization at all** (not even Whisper's English normalizer, unlike the
  ASR-Finetune pipeline), and **no CER**. *(Verified artifact.)*
- The train/test split happens entirely **outside this repository** -- the
  notebook only loads pre-existing `data/train/` and `data/test/` folders from
  a `data.zip` the user is expected to have already prepared externally. No
  split-construction code is committed here at all, so unlike the two
  previously reviewed pipelines this repo's own code cannot be directly
  checked for the speaker-leakage bug -- but there is also no evidence
  anywhere in what *is* committed that speaker identity was ever tracked or
  considered. *(Unknown, not confirmed either way, unlike the other two.)*
- The metadata-generation script (`audio_folder_creation.py`) crudely escapes
  commas and quotes in transcripts by replacing them with spaces rather than
  proper CSV quoting -- a real, if minor, data-corruption risk for any
  transcript containing those characters. *(Verified artifact.)*

## Result

**52.02% WER** after 1000 steps (from the committed screenshot,
`img/fine_tuning_eval.png`, showing the notebook's own training-time eval
table). *(Recorded result -- the image is a real screenshot of that run's
output, not independently reproduced here.)*

## Can it be verified?

Not from what's committed: no saved predictions, no split-construction code,
no normalization detail beyond raw `jiwer`. The number is real in the sense
that it is a genuine screenshot of a real training run's own logged eval
metric, but it cannot be checked for split integrity or reproduced from this
repository alone.

## Why this is still useful evidence, carefully qualified

Despite thinner rigor and far less data than either this project's own runs
or the two previously reviewed external pipelines, this run's full-parameter
fine-tune reached 52.02% WER in **1000 steps on a single OpenSLR shard** --
notably better than this project's own LoRA ceiling (78-88% WER across
E004-E006, on up to 100 hours and thousands of steps). This is now the
**third independent codebase**, alongside the leakage-tainted historical
17%-WER full-parameter checkpoint and SPEAK-ASR's Optuna-tuned 10.85%
whisper-medium result, showing the same qualitative pattern: full-parameter
fine-tuning appears to reach materially lower WER than LoRA on this task, even
under far less rigor and far less data.

None of these three, individually, is trustworthy proof -- two have confirmed
or suspected evaluation-leakage problems, and this one's split is simply
unverifiable either way. But three independently-run pipelines landing on the
same qualitative direction, via different code, different data slices, and
different (or absent) evaluation rigor, is a stronger signal than any one of
them alone. It does not tell us *how low* LoRA's true ceiling is relative to
full fine-tuning on this project's own honest, speaker-disjoint evaluation --
only that the direction is consistent enough to weigh in scoping the
already-queued full-parameter pilot.
