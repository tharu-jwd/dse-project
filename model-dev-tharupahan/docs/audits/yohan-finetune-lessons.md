# Yohan's Whisper-small Sinhala fine-tuning: lessons for our own full fine-tune

Mined 2026-09-10 from Yohan's four branches on `github.com/tharu-jwd/dse-project`
(`Yohan_Finetune`, `Yohan_Observation`, `Yohan_augmentation`, `Yohan_RealTime`),
the HF repo [`Yohan2003/whisper-small-sinhala`](https://huggingface.co/Yohan2003/whisper-small-sinhala)
(model weights + data splits + training logs), and the dataset repo
[`Yohan2003/whisper-sl-data`](https://huggingface.co/datasets/Yohan2003/whisper-sl-data).

**Standing constraint (unchanged):** per
[the comparison-only memory](../../../../.claude/projects/-Users-tharupahan-Code-dse-project/memory/yohan-checkpoints-comparison-only.md),
Yohan's checkpoints are a comparison baseline only, never a training starting
point. This document is about *not repeating his mistakes*, not adopting his
artifacts.

## His runs, at a glance (from `final-scripts/finetune_tracker.csv`)

| Run | Type | LR / sched | Eff. batch | Data | Val WER/CER | Test WER/CER | English WER Δ vs base | Forgetting |
|---|---|---|---:|---|---|---|---:|---|
| `run1-lr3e-5-bs32` | full | 3e-5 / linear | 32 | stratified v1 | 19.96 / 4.17 | 17.08 / 3.49 | **+76.59 pts** | **severe** |
| `run6-lr3e-5-bs32_amd` | LoRA (q,v) | 5e-5 / linear | 64 | v1 | 23.81 / 6.46 | 21.01 / 5.73 | **+69.47 pts** | **severe** |
| `run2-lr3e-5-bs32_lora` | LoRA | 3e-5 / cosine | 32 | v1 | 61.63 / 18.81 (crash) | — | — | interrupted @ ep 2.5 |
| `run3-lr1e-4-r32-lora` | LoRA (q,k,v,out,fc1,fc2) | 1e-4 / cosine | 32 | v1 | 26.01 / 7.26 | 25.99 / 7.06 | +2.76 pts | mild |
| `full-lr1e-5-e4-cosine-bs64` | full | **1e-5 / cosine** | 64 | stratified v2 | 28.38 / 8.03 | 28.47 / 7.93 | **+1.79 pts** | **mild** |
| `run5-v4-lr3e-5-bs64-v3-resume` (aka run7) | full | 3e-5 / linear | 64 | stratified v4 | 22.41 / 6.65 | **19.06 / 4.90** | not measured (skipped) | unknown |

`run5`/`run7` is the one that prompted this review ("he has done a new one and
has 19% in it"). Model weights: `models/run7-v4-lr3e-5-bs64/model.safetensors`
on the HF repo, uploaded 2026-09-09.

## 1. English catastrophic forgetting -- the finding that changes our plan

Yohan measured English retention properly (LibriSpeech test-clean, 2,620 rows,
`evaluate_english_forgetting.py`) on five runs. The pattern is stark:

- **Full fine-tune at lr=3e-5, linear schedule, no mitigation -> English WER
  4.27% -> 80.86% (+76.59 pts). Severe.** A full fine-tune at this LR
  essentially destroys English.
- **LoRA at lr=5e-5 with narrow targets (q_proj, v_proj only) -> +69.47 pts.
  Also severe.** Narrow LoRA + high LR does not protect English.
- **LoRA at lr=1e-4, cosine, WIDE targets (q,k,v,out_proj,fc1,fc2) -> +2.76
  pts. Mild.**
- **Full fine-tune at lr=1e-5, cosine schedule -> +1.79 pts. Mild -- and
  *better* than the wide-LoRA run despite being a full fine-tune.** Low LR +
  cosine limited drift from the base model.

**Consequence for us:** E012's search picked **lr=5e-5** as the Sinhala-
convergence winner at 100-step pilot scale. That is *higher* than the lr=3e-5
that caused severe forgetting in Yohan's hands. Our teacher-behaviour replay
(which Yohan does not use) may rescue it -- but that is exactly what E014 has
to measure, not assume. **E014 must include an lr=1e-5 + cosine arm**, not
just lr=5e-5, and the retention gate on the winning arm is non-negotiable
before any full run.

Mechanical note: `finetune_whisper.py` / `finetune_whisper_lora.py` hardcode
`model.generation_config.language = "sinhala"` into the saved checkpoint. The
English retention eval must force `language="english"` at generation
time -- otherwise you are testing "transcribe English while the model is told
it's Sinhala", a harsher and less informative measurement.

## 2. Per-epoch generation eval dominates GPU time

Measured from `run7`'s own training log on an RTX 4090:

- Training: ~1.1 s/optimizer-step, 7,704 steps for 4 epochs -> **~8 h of pure
  training** (~2 h/epoch), effective batch 64 (per-device 8 x grad-accum 8),
  ~123k train rows.
- **Per-epoch generation-based validation on 15,763 rows at eval batch 4 =
  ~84 minutes each** (`eval_runtime: 5062 s`). Four epoch-end evals ~= 5.6 h.
- **Eval was ~40% of total GPU time.**

Our v5 validation set is 20,232 rows -- even larger. **Do per-epoch eval on a
small fixed subset (e.g. the 200-row slice already built for the E012/E014
pilots), and run the full v5 validation once at the very end.** Otherwise a
4-epoch full run spends a third of its budget re-decoding the same validation
set.

## 3. OOM traps (full fine-tune Whisper-small on a 24 GB card)

- Per-device train batch 16 OOM'd; batch 64 "OOM'd hard". `run7` needed
  **`--per-device-train-batch-size 8` + `--per-device-eval-batch-size 4`**
  with `--gradient-accumulation-steps 8` for an effective batch of 64.
- **Generation-based eval is the OOM risk, not training** -- that's why the
  eval batch had to drop to 4 while the train batch stayed at 8.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment was
  needed to avoid fragmentation-related OOM on resume.
- Per-device batch 32 (no accumulation) needs far more memory per step than
  8 x grad-accum-4 for the *same* effective batch -- always use gradient
  accumulation on <=24 GB rather than a large per-device batch.
- An AMD MI300X (192 GB) fit batch 64 with no accumulation, but that's exotic
  hardware; don't plan around it.

## 4. Dependency version hell

Yohan's `final-scripts/requirements.txt` comments document each pin as a scar.
The traps that a naive `pip install -r` does **not** protect against (pip
won't touch an already-installed version that technically satisfies the
constraint):

- **`transformers` 4.41-4.45** satisfies `>=4.41` but crashes at runtime:
  `Seq2SeqTrainingArguments(eval_strategy=...)` -- `eval_strategy` (replacing
  `evaluation_strategy`) was only added in **4.46.0**. `TypeError: unexpected
  keyword argument 'eval_strategy'`.
- **`accelerate` 0.30-0.34** satisfies `>=0.30` but crashes at
  `trainer.train()`: transformers 4.57.x calls
  `Accelerator.unwrap_model(keep_torch_compile=...)`, a kwarg only in
  accelerate **1.x**.
- **`numpy>=2`** silently breaks `pyarrow` / `soundfile` / `librosa` -- wheels
  built against numpy 1.x segfault or ABI-mismatch against numpy 2.x arrays.
  Pin **`numpy<2.0.0`**.
- **Resuming a checkpoint across transformers versions** (`run5`/`run7`'s
  actual blocker): a checkpoint saved by `transformers==4.57.6` then loaded on
  a fresh pod broke two ways -- (1) a `torch.load` security check
  (CVE-2025-32434) refused to load `optimizer.pt` / `scheduler.pt` without
  `torch>=2.6`, and (2) `trainer_state.json` had a `best_global_step` key that
  older `TrainerState` schemas reject. Fix used: pin
  **`transformers==4.46.3`** (below the `torch.load` restriction, still has
  `eval_strategy`) and **strip `best_global_step` from `trainer_state.json`**
  before resuming.

**Rule: pin every version from one mutually-tested file, install them all at
once, and smoke-test the actual training entry point** -- a clean
`pip install -r` is not evidence the run will start.

## 5. Checkpoint / ephemeral-pod discipline

- `run3` was interrupted **three times** by "GPU pod switches" (started on an
  RTX 2000 Ada, moved to two different RTX 4090 pods). Pod-local disk is
  ephemeral.
- Checkpoint every epoch, and **copy the checkpoint off the pod before
  terminating it** (`runpodctl send` works pod->local too). `run5`/`run7` was
  deliberately paused after epoch 3 via `pkill` right after the checkpoint
  save was confirmed complete, then resumed the next day on a new pod.
- `finetune_whisper.py` grew `--resume-from-checkpoint` specifically for this
  -- restores model + optimizer + scheduler + RNG state and continues from the
  saved `global_step`.
- `load_best_model_at_end=True` + `metric_for_best_model="wer"` means the model
  saved to the top-level output dir is **whichever epoch had the best
  validation WER, not necessarily the last checkpoint**. Always know which
  checkpoint a reported number was computed against.

## 6. Data-quality issues that cap achievable WER

Yohan's own error analysis (`final-scripts/eval_results/error_analysis/README.md`)
is explicit that these are label problems -- "this is a data-labeling
consistency issue, not a model-capacity issue -- it will not go away with more
training on the same data":

- **Word-boundary / compounding inconsistency in the ground truth itself.**
  The same compound is written as one token in some rows and two tokens in
  others, *within the same split* (`ටීවිවල` vs `ටීවි වල`, `බොරුකීම` vs `බොරු
  කීම`). This is the single largest error cluster in nearly every one of his
  runs. The four source corpora (OpenSLR-52, YouTube, BizBrains, Linga) don't
  share a transcription/compounding standard.
- **Spoken vs written register mismatch.** The #1 substitution pattern in
  every run is a spoken-register word "corrected" to/from its formal
  equivalent: `කියල<->කියලා`, `කරන්නෙ<->කරන්නේ`, `සමග<->සමඟ`, `කල<->කළ`,
  `කථා<->කතා`. OpenSLR-52 (~97% of his training pool) is read/formal speech;
  the YouTube/BizBrains/Linga slices are conversational. The model hedges
  between conventions instead of reproducing whichever is in front of it.
- **ZWJ / conjunct-consonant encoding inconsistency.** Sinhala conjuncts
  (`යන්සය`, `රකාරාංශය`) can be Unicode-encoded with or without an explicit
  ZWJ; different keyboards/tools normalize differently. The recurring
  `්‍ය` / `‍ය` / `්‍ර` / `‍ර` error clusters were ~11% of `run1`'s wrong
  samples. **Recommendation from his own analysis: NFC / ZWJ-normalize all
  transcripts before training -- a cheap one-time data fix.**
- **Hidden zero-width characters** (from `scripts/clean_hidden_chars.py`):
  **ZWNJ (U+200C) is always noise in Sinhala** orthography (unlike
  Devanagari/Bengali) -- strip unconditionally. **ZWJ (U+200D)** is only valid
  immediately between a virama (U+0DCA) and a Sinhala consonant (a real
  conjunct like `ශ්‍රේෂ්ඨ`) -- strip it everywhere else.
- **Linga source is byte-for-byte duplicate audio of OpenSLR-52** (identical
  MD5 on the raw WAV bytes -- same recording under two `source_dataset`
  labels). His `dedup_collection_against_openslr.py` dedups on **audio-byte
  hashes, not text**: text dedup gives false positives on shared prompt
  sentences read by different speakers, and false negatives across
  transcript-cleaning differences.
- **Single-particle deletions.** The top "deleted words" in `run5` are 1-char
  Sinhala grammatical particles (`ය`, `ම`, `ද`, `ව`, `නම්`, `නේ`). The model
  drops them, inflating WER on otherwise-correct transcriptions.
- **Underrepresented religious/honorific vocabulary** (`වහන්සේ` and the
  monastic register) -- garbled across runs; likely Dhamma-talk content that's
  only ~2% of the training data.

## 7. Evaluation-comparability traps

- **`run5`/`run7`'s test WER (19.06%) came out *better* than its validation
  WER (22.41%)** -- Yohan's own tracker note flags this as unusual ("unlike
  other runs in this tracker where val and test track closely"). Possible test
  contamination, an easier test distribution, or a split artefact.
- **The split scripts in the repo are source-proportional random splits, with
  no speaker-grouping logic anywhere in the code** (`scripts/split_dataset.py`,
  `scripts/split_final_datasets.py` -- both stratify only on `source_dataset`
  via `sklearn.train_test_split`). `stratified_v4` is *described* as
  "speaker-disjoint" in `finetuneGuide.md` and `RESUME_NOTES.md`, but that
  claim cannot be verified from the available code. If v4 is not genuinely
  speaker-disjoint, the 19.06% is leakage-inflated. (This matches our own
  [historical audit](historical-audit.md): prior team results near 17% WER
  used a random row-level split with speaker leakage.)
- **His WER normalization** (`jiwer.Compose([ToLowerCase, RemovePunctuation,
  RemoveMultipleSpaces, Strip])`) strips *all* punctuation and lowercases --
  far more permissive than our conservative Sinhala-specific
  `metric_normalize`. His numbers are not on the same scale as ours.
- **Run names lie.** `run6-lr3e-5-bs32_amd` "is actually 5e-5" per its own
  tracker note. Augmentation prob/range values weren't logged to W&B ("assumed
  unless this run passed override flags"). Read the actual logged
  hyperparameters, never the run name.

## 8. Model-behaviour findings

- **Degenerate repetition loop.** The worst individual samples in his LoRA runs
  are the decoder stuck repeating one word ~40 times for a short reference
  (`මෙම මෙම මෙම ...`), on acoustically hard / noisy / very short input. It
  shows up in the **LoRA** runs, **not** the full fine-tune (`run1`) -- a full
  fine-tune updates all weights and is more robust at confidently terminating
  generation. Add `no_repeat_ngram_size=3` (or a repetition penalty) at
  generation time regardless; our `train.py` already does this for eval.
- **Full fine-tune keeps improving every epoch through 4** (`run1` shows no
  plateau; `run5` val WER 31.12 -> 26.23 -> 23.62 -> 22.41). **One epoch
  undertrains a full fine-tune** -- our historical one-effective-epoch LoRA
  convention does not transfer; plan for 3-4 epochs.
- `run3` (LoRA "best-epoch2") was a clear outlier (90% of samples wrong, error
  shape unlike the others) -- Yohan's analysis concludes it was
  undertrained/misconfigured/a checkpoint mismatch, not representative of
  LoRA's ceiling. Lesson: verify the checkpoint you evaluate is the one you
  intended.

## Actionable checklist for our full fine-tune

- [ ] E014: add an **lr=1e-5 + cosine** arm alongside lr=5e-5; both scored
      against the frozen 2,620-row LibriSpeech benchmark. Do not scale to a
      full run on an unvalidated LR.
- [ ] Per-epoch eval on a **small fixed val subset**; full v5 validation once
      at the end only.
- [ ] Train batch 8 / eval batch 4 / grad-accum to the target effective batch;
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [ ] One pinned `requirements.txt`; smoke-test the training entry point, not
      just `pip install -r`. For any resume, pin `transformers==4.46.3` and be
      ready to strip `best_global_step` from `trainer_state.json`.
- [ ] Checkpoint every epoch; copy off ephemeral pods before terminating.
- [ ] Before training: NFC/ZWJ-normalize all transcripts; strip ZWNJ
      unconditionally and context-invalid ZWJ; confirm no audio-byte duplicates
      across sources; decide and enforce one compounding/spacing convention.
- [ ] Plan for **3-4 epochs**, not one.
- [ ] Keep our own conservative `metric_normalize` for all reported numbers;
      do not compare directly against Yohan's punctuation-stripped WER.
