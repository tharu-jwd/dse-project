# Review: fine-tuning/evaluation work against the SRS and Software Architecture Document

I read `project-docs/SRS.pdf` and `project-docs/Software Architecture.pdf` cover to cover and
checked `model-development/` against them. Scope here is strictly fine-tuning and evaluation,
the two workflows this folder owns. Preprocessing, the backend, and the frontend aren't covered.

This is an independent read, not a rubber stamp. Where the SRS/SAD say something specific and
the code doesn't do it, that's a gap. Where I think a documented choice is questionable on its
own merits, that's a judgment call instead, and I say so. Each item below ends with what I'd
actually do about it, not just the finding.

## Critical gaps

**1. Wrong base model default.**
SRS §3.1.2 and SAD §3.2/§6/§10 say Whisper-Small, fallback to Whisper-Base, tied directly to the
Colab Pro GPU memory limit. `train_asr.py --base-model` defaults to `openai/whisper-medium`,
about 3x the size the SRS says the hardware rules out.

*My call: fix it.* Change the default to `openai/whisper-small` and update the README and
`asr_common/models.py` docstrings that assume medium. `--base-model` already exists as a flag,
so this is a one-line change plus a docs pass, nothing structural. No reason to sit on this one.

**2. No CER anywhere.**
WER and CER are required together throughout both docs. `asr_common/metrics.py` only has
`compute_wer`. `jiwer.cer` exists and takes the same arguments as `jiwer.wer`, so there's no
library gap here, just a missing function.

*My call: fix it.* Add `compute_cer` next to `compute_wer`, wire it into `train_asr.py`'s
`compute_metrics` and `evaluate_asr.py`'s scoring, and add a `cer` field to the persisted
`eval_results/results.jsonl` rows. Maybe twenty minutes of work. No excuse to skip it.

**3. No Weights & Biases integration.**
Called out as a defined term, an explicit pipeline step (SAD Figure 5), a required software
interface with protocol and auth spelled out, and a project dependency. `train_asr.py
--report-to` defaults to `"none"`, `wandb` isn't a dependency, nothing logs to it.

*My call: fix it, and it's cheaper than it looks.* `Seq2SeqTrainer` already reports loss,
learning rate, and `compute_metrics` output to W&B automatically once `report_to="wandb"` is
set, no custom instrumentation needed. Add `wandb` to `requirements.txt`, change the default (or
at least document `--report-to wandb` clearly as the real-run setting), done. The smoke test can
keep `--report-to none` since there's no reason to log throwaway 5-step runs.

**4. No SpecAugment.**
SRS §3.1.2 requires it by name during training. `train_asr.py` goes straight from audio to
features with no masking step.

*My call: fix it, and it's not a custom implementation.* `WhisperConfig` already has
`apply_spec_augment` (off by default) plus `mask_time_prob`/`mask_feature_prob` knobs built into
the encoder's forward pass. Setting `base_model.config.apply_spec_augment = True` before wrapping
with LoRA gets this for free. Worth a quick check that PEFT-wrapping doesn't disable it, but I'd
be surprised if it does.

**5. No best-checkpoint selection by validation metric.**
SRS §3.1.2: pick the checkpoint with the best validation WER/CER, not the last one.
`Seq2SeqTrainingArguments` never sets `load_best_model_at_end`.

*My call: fix it.* Add `load_best_model_at_end=True`, `metric_for_best_model="wer"` (or `"cer"`
once #2 is in), `greater_is_better=False`. Three lines. Do this at the same time as #2 since
they touch the same training arguments.

## Notable gaps

**6. Only one external baseline is wired in.**
SRS §4.1 names two published Sinhala fine-tunes to benchmark against: SPEAK-ASR (already used)
and `Lingalingeswaran/whisper-small-sinhala`, never referenced here. Same name shows up on
`Yohan_Observation` as a dataset notebook (`lingaData.ipynb`), so it's worth confirming with
whoever owns that branch whether the dataset and the model are actually related or just share a
name.

*My call: check the model card first, then decide.* If it's a LoRA adapter, `--model custom
--checkpoint Lingalingeswaran/whisper-small-sinhala` probably already works with what's built.
If it's a merged model, see #7. Either way this is a five-minute lookup before deciding whether
any code needs to change.

**7. `--model custom` assumes a PEFT adapter.**
It always calls `PeftModel.from_pretrained`. A fully merged model would fail to load through it.

*My call: don't build this until #6 tells us we need it.* Speculatively adding a code path for a
model format we haven't confirmed exists yet is wasted work. If Lingalingeswaran's model turns
out to be merged, add a `--model-format {adapter,full}` flag or similar then, not now.

**8. No separate generalization-eval split.**
The SRS wants two numbers: WER/CER on a held-out split from the pooled training sources, and
separately on the fully-withheld Mozilla Common Voice set. Right now there's one hardcoded
dataset.

*My call: leave it.* This is downstream of preprocessing delivering the real pooled dataset,
which doesn't exist yet. Building support for a split that doesn't exist against data we don't
have is premature. Once preprocessing lands both sources, running `evaluate_asr.py` twice with
two different `--dataset`/`--split` values covers this without any code change, it's a usage
pattern, not a missing feature.

**9. No confusion-pattern / error-analysis output.**
SRS §3.1.3 wants a breakdown of frequently mistranscribed words or phonemes. The SAD splits this
into a separate use case (UC-11, "Development Team") from evaluation and benchmarking (UC-10,
"Developer"), so ownership is genuinely unclear.

*My call: defer.* This is much more useful run against a real trained checkpoint than the smoke
test, and it's arguably not solely mine to build given the SAD's own use-case split. I'd revisit
after the first real fine-tuning run produces something worth analyzing, not before.

**10. No aggregate output report.**
SRS §3.1.3 wants a summary report regardless of outcome. We have `eval_results/results.jsonl`,
raw rows, no rendered summary.

*My call: low priority, do it once there's real data.* A script that reads the JSONL and spits
out a markdown table is maybe half an hour of work, but there's nothing meaningful to summarize
until actual runs exist beyond the smoke test. Not worth building against fake data.

## Judgment calls

**11. LoRA vs. full fine-tuning.**
I used LoRA, matching SPEAK-ASR's own approach. Neither document mentions LoRA, PEFT, or
adapters anywhere, not even in the glossary, where `SpecAugment`, `FP16`, and `Checkpoint` all
get entries. The SRS's cited fine-tuning reference is the well-known Hugging Face blog post on
full fine-tuning, not LoRA. Put together, the original intent probably was full fine-tuning of
Whisper-Small.

*My call: keep LoRA, but this needs a real conversation, not a silent decision either way.* LoRA
is almost certainly the better engineering choice given the same GPU constraints the docs
themselves cite, and it keeps the SPEAK-ASR comparison apples to apples, adapter versus adapter.
But it's a real deviation from what's referenced, and if the team or mentor wants literal
compliance, that means full fine-tuning of Whisper-Small instead, a noticeably more expensive
setup than what's here now. I'm not switching this without someone else weighing in.

**12. The placeholder dataset doesn't match the SRS's real training-data shape.**
SRS §3.1.1/§3.1.2 wants OpenSLR52, YouTube-Sinhala-ASR, and Biz-Brains-Academy-Sinhala pooled for
train/validation, with Mozilla Common Voice held out entirely for eval. `dev_dataset.py` uses
only `SPEAK-ASR/openslr-sinhala-asr`, OpenSLR alone, not the four-way split.

*My call: no action, this is working as intended.* This folder doesn't own dataset pooling or
the held-out-set strategy, preprocessing does, and the placeholder is already documented as
temporary. Flagging it here purely so whoever builds preprocessing knows the real target shape:
two distinct sources, a pooled three-way train/val set and a fully separate held-out Mozilla
Common Voice set, not the one dataset standing in for both right now.

## What's actually correct

- WER as a metric, the prediction-collection approach, and treating SPEAK-ASR as an external
  comparison point rather than something being rebuilt all match the SRS/SAD. CER is missing,
  but what's there is implemented correctly.
- The CLI-driven, fully-parameterized hyperparameters already support running "at least three
  distinct configurations" (SRS §3.1.2) with no code changes, just different flags per run.
- Keeping preprocessing out of scope and isolating every placeholder-dataset assumption in one
  file matches the SRS's own split between §3.1.1 and §3.1.2/§3.1.3, and matches what's actually
  being built on `Yohan_Observation` (dataset notebooks for the same three sources the SRS
  names).
- A PEFT adapter directory satisfies the SRS/SAD's generic checkpoint definition. Neither
  document mandates a specific serialization format.
- `eval_results/results.jsonl` writes with `ensure_ascii=False`, so it won't mangle Sinhala text
  if any ends up in a logged field later.

## Not reviewed here

VAD precision/recall against a manually labelled subset (SRS §3.1.3) is a preprocessing task,
not this folder's. Backend `Transcriber` integration is covered in
[`INTEGRATION_POINTS.md`](INTEGRATION_POINTS.md), not here. I also checked the ERD directly:
nothing in it relates to fine-tuning or evaluation, it's entirely the Phase 2 application's
schema (users, media, transcripts, quizzes), no checkpoint or experiment-tracking tables.

## If I were prioritizing the fix list

Do #2, #4, #5 together first, they're all small `Seq2SeqTrainingArguments`/`metrics.py` changes
and touch the same code path. #1 is a one-line default plus a docs pass, do it whenever, doesn't
block anything else. #3 needs the `wandb` dependency added and a decision on the default
`--report-to` value, otherwise it's also small. #6 is a five-minute lookup before deciding
anything. #11 needs a mentor/team conversation before either LoRA or full fine-tuning gets locked
in for a real run. Everything else waits for either the real preprocessing pipeline or an actual
trained checkpoint to exist.
