# Review: fine-tuning/evaluation work against the SRS and Software Architecture Document

I read the project's `project-docs/SRS.pdf` and `project-docs/Software Architecture.pdf` in full (not just
skimmed) and checked what's in `model-development/` against them. This is scoped strictly to
fine-tuning and evaluation, the two workflows this folder owns. It does not cover preprocessing,
the backend, or the frontend.

This is an independent read. Where the SRS/SAD state something specific and the code doesn't
match it, that's flagged as a gap. Where I think a documented choice is questionable on its own
technical merits, that's flagged separately as a judgment call, not a defect.

## Critical gaps: explicit requirements the code doesn't meet

**1. Wrong base model default.** SRS §3.1.2 and the SAD (§3.2, §6, §10) specify Whisper-Small as
the primary architecture for fine-tuning, with Whisper-Base as a fallback if GPU memory is
insufficient, tied directly to the stated hardware constraint (Google Colab Pro, free-tier
fallbacks). `train_asr.py --base-model` currently defaults to `openai/whisper-medium`, and every
mention of the base model in `model-development/README.md` and `asr_common/models.py` assumes
medium. Medium is roughly 3x the parameter count of Small and was exactly the size the SRS says
the hardware constraint rules out as a default. `--base-model` is already a CLI flag, so this is
a one-line default change, not a redesign, but it's a real mismatch as it stands.

**2. No CER anywhere.** WER and CER are required together throughout both documents (SRS §1.3
definitions, §3.1.2, §3.1.3, §3.3 accuracy criterion, §3.9.3, §3.11; SAD §6 process view, UC-10).
`asr_common/metrics.py` only implements `compute_wer`. There's no `compute_cer`, and neither
`train_asr.py`'s `compute_metrics` nor `evaluate_asr.py`'s scoring loop reports it.

**3. No Weights & Biases integration.** W&B logging of training loss, validation loss, learning
rate, WER, and CER is called out repeatedly and specifically: it's a defined term (SRS §1.3), an
explicit step in the SAD's fine-tuning activity diagram (Figure 5: "Log metrics to Weights &
Biases"), part of UC-9's description, a listed software interface (SRS §3.9.3, with protocol,
auth, and payload format specified), a purchased component (SRS §3.8), and a project dependency
assumption (SRS §2). `train_asr.py --report-to` defaults to `"none"`, `wandb` isn't in
`requirements.txt`, and there's no W&B code anywhere in this folder.

**4. No SpecAugment.** SRS §3.1.2: "SpecAugment is applied during training to reduce
overfitting." `SpecAugment` is a defined term in both documents' glossaries. `train_asr.py` has
no augmentation step; `build_preprocess_fn` goes straight from raw audio to log-Mel features with
nothing masked.

**5. No best-checkpoint selection by validation metric.** SRS §3.1.2: "the checkpoint with the
best validation WER/CER is selected rather than simply using the last epoch." `train_asr.py`'s
`Seq2SeqTrainingArguments` sets `save_strategy`/`save_total_limit` (a rolling window of recent
checkpoints) but never sets `load_best_model_at_end=True` or `metric_for_best_model`, so nothing
about the current setup actually picks the best-scoring checkpoint over the last one.

## Notable gaps: real, but lower severity or shared ownership

**6. Only one external baseline is wired in.** SRS §3.1.3 and §4.1 name two existing published
Sinhala Whisper fine-tunes to benchmark against: SPEAK-ASR (already used, `--model speak-asr`)
and `Lingalingeswaran/whisper-small-sinhala` (SRS ref [9], not referenced anywhere in this
folder). Worth noting: that name also showed up on the `Yohan_Observation` branch as a dataset
notebook (`lingaData.ipynb`, the "lingalingeswaran-asr-sinhala-dataset"), so there may be a
naming overlap between a dataset and a model worth double-checking with whoever owns that branch
before assuming they're the same source.

**7. `evaluate_asr.py --model custom` assumes a PEFT adapter.** It routes every checkpoint
through `load_lora_adapter`, which calls `PeftModel.from_pretrained(base_model, adapter_id)`.
If `Lingalingeswaran/whisper-small-sinhala` (or any other benchmark target) turns out to be a
fully merged model rather than a raw adapter, this path won't load it. Not fixable without
knowing that model's actual format.

**8. No separate "generalization" evaluation split.** SRS §3.1.3 wants two evaluation numbers:
WER/CER on a held-out test split drawn from the pooled training sources, and separately on the
fully-withheld Mozilla Common Voice Sinhala set, specifically to measure generalization to an
unseen source. `evaluate_asr.py` only knows about one dataset/split at a time
(`SPEAK-ASR/openslr-sinhala-asr`, hardcoded). This is expected for now since it's explicitly a
placeholder (see "Placeholder dataset" in the README), but the eventual real setup will need two
distinct eval datasets, not one, and the current single-`--split` design doesn't yet anticipate
that.

**9. No confusion-pattern / error-analysis output.** SRS §3.1.3: "a confusion-pattern breakdown
is produced to identify frequently mistranscribed phonemes or words." The SAD's use-case table
attributes this to "Analyse recognition errors" (UC-11, "Development Team"), separate from
"Evaluate and benchmark model" (UC-10, "Developer"), so it's ambiguous whether this belongs to
the evaluation script specifically or is a broader, shared analysis task. Either way,
`evaluate_asr.py` doesn't produce anything beyond aggregate WER and per-sample REF/HYP text.

**10. No aggregate output report.** SRS §3.1.3: "A final output report summarising all
evaluation results is produced regardless of whether the model clears the accuracy threshold."
`eval_results/results.jsonl` accumulates one row per run, which covers the "regardless of
outcome" part, but there's no generated report artifact (markdown, HTML, whatever) summarizing
results across runs.

## Judgment calls: places I'd push back on, or flag as open questions

**11. LoRA vs. full fine-tuning.** I chose LoRA, following the SPEAK-ASR baseline's own approach.
Neither document mentions LoRA, PEFT, or adapters anywhere, including in the glossary, where
`SpecAugment`, `FP16`, and `Checkpoint` all get defined. The SRS's fine-tuning reference (§1.4
ref [5], "Fine-Tune Whisper for Multilingual ASR with Transformers") is the well-known Hugging
Face blog post that does full fine-tuning, not LoRA. Read together, that suggests the original
intent was probably full fine-tuning of Whisper-Small.

My own judgment: LoRA is very likely still the better call here regardless of what was
originally intended, given the explicitly documented hardware constraint (Colab Pro / free-tier
GPU memory and session limits) and because it keeps the comparison against SPEAK-ASR
methodologically consistent (adapter vs. adapter, not adapter vs. full fine-tune). I'd raise this
with the team and mentor as a deliberate deviation to confirm rather than silently "fix" it
either direction. If the team wants literal compliance with the referenced blog post, that means
full fine-tuning of Whisper-Small, a materially different and more expensive training setup than
what `train_asr.py` currently does.

**12. The placeholder dataset doesn't match the SRS's real training-data shape, on purpose.**
SRS §3.1.1/§3.1.2 describes pooling OpenSLR52 + YouTube-Sinhala-ASR + Biz-Brains-Academy-Sinhala
for train/validation, with Mozilla Common Voice Sinhala withheld entirely for evaluation.
`asr_common/dev_dataset.py` uses only `SPEAK-ASR/openslr-sinhala-asr` (OpenSLR data alone,
already split by SPEAK-ASR, not the four-dataset pool). This is intentional and already
documented as temporary: this folder doesn't own dataset pooling, cleaning, or the held-out-set
strategy, that's the preprocessing pipeline's job per the scope agreement already in place. Not
a defect in this folder's work. Flagging only so whoever picks this up next knows the real target
shape once preprocessing is ready: two distinct sources (pooled 3-way train/val, and a
completely separate held-out Mozilla Common Voice set), not the single dataset currently
standing in for it.

## What's actually correct

- WER as a metric, the prediction-collection approach, and scoring the SPEAK-ASR baseline as an
  external comparison point (not something being rebuilt) are all aligned with the SRS/SAD.
  CER is missing, but what's there is correctly implemented.
- The CLI-driven, fully-parameterized hyperparameters in `train_asr.py` support running "at
  least three distinct configurations" (SRS §3.1.2) without any code changes, just different
  flags per run.
- Treating preprocessing as out of scope and isolating every placeholder-dataset assumption in
  one file matches the SRS's own separation between §3.1.1 (preprocessing) and §3.1.2/§3.1.3
  (fine-tuning/evaluation), and matches what's actually being built on the `Yohan_Observation`
  branch (dataset notebooks for the same three sources named in the SRS).
- The `Checkpoint` output format (a PEFT adapter directory) satisfies the SRS/SAD's generic
  "saved snapshot of a model's trained parameters" definition; nothing in either document
  requires a specific serialization format.
- `eval_results/results.jsonl` uses `ensure_ascii=False`, so it won't mangle Sinhala text if any
  ever ends up in a logged field.

## Not reviewed here (out of this folder's scope)

VAD precision/recall against a manually labelled subset (SRS §3.1.3) is a preprocessing-pipeline
evaluation task (VAD is part of §3.1.1), not this folder's job. The backend `Transcriber`
integration is covered in [`INTEGRATION_POINTS.md`](INTEGRATION_POINTS.md), not here. I also
checked the ERD directly and didn't find anything relevant to fine-tuning or evaluation in it,
it's entirely the Phase 2 application's schema (users, media, transcripts, quizzes), with
nothing for checkpoints or experiment tracking.
