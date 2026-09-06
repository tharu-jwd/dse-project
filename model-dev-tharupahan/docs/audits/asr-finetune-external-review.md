# External Review: SPEAK-ASR/ASR-Finetune

Evidence-graded review of `/Users/tharupahan/code/ASR-Finetune` (GitHub
`SPEAK-ASR/ASR-Finetune`), a separate, more recently active Sinhala Whisper
fine-tuning codebase in the same broader project ecosystem (same `SPEAK-ASR`
org referenced elsewhere in this project, e.g. the backend's transcriber
configuration). Not the `Yohan_*` work already covered in
[the historical audit](historical-audit.md) -- a different, later codebase,
reviewed the same way: read the actual code and data, not just claims.

Evidence labels follow the same convention as the historical audit:
**Verified artifact** (reproducible from committed code or a fetched model
card), **Recorded result** (present in a HF model card, run and checkpoint not
independently reproduced here), **Interpretation** (conclusion drawn from
verified artifacts), **Unknown**.

## What they built

A substantially more mature ML-engineering pipeline than this project's
inherited `Yohan_*` scripts: dataclass-based configuration for every concern
(dataset, model, LoRA, training, Hub), Weights & Biases experiment tracking,
continuous checkpoint sync to the HF Hub (`hub_strategy="checkpoint"`),
streaming dataset support (a later branch), an Optuna-based automated
hyperparameter search (`hyper-pram-tuning` branch, 421-line optimizer,
`direction="minimize"` on `eval_wer`), and an experimental pseudo-labeling /
self-training stage (`new-pipeline` branch, `collect_pseudo_data.py` +
`pretrain_postproc.py`, appears exploratory/incomplete). *(Verified artifact.)*

Compute: `vast_startup.sh` / `.vastai/README.md` on several branches confirm
paid, rented GPUs via [Vast.ai](https://vast.ai/), not free-tier Colab/Kaggle.
`bf16` training and batch sizes of 32-128 across "4 GPUs" (per one model card)
are only feasible on real Ampere-class-or-newer rented hardware. This is a
materially larger compute budget than this project's free-tier constraint.
*(Verified artifact + Recorded result.)*

## Data

Same broad source families as this project (OpenSLR Sinhala + YouTube
Sinhala), but sourced through a third-party HF re-upload,
`irudachirath/large-sinhala-asr-dataset`: **185,293 rows**, two columns only
(`text`, `audio`), no speaker ID, no split, essentially undocumented
provenance (empty dataset card). *(Verified artifact -- fetched directly.)*

That row count is the same 185,293 this project's own independently-built v4
manifest totals across train/validation/test/heldout/excluded. This is strong
circumstantial evidence it is the same underlying audio, repackaged by a third
party with speaker identity stripped -- not proof of byte-identical content,
but consistent with it. *(Interpretation.)*

Four resulting HF datasets get combined for training:
`SPEAK-ASR/openslr-sinhala-asr-preprocessed-{1,2,3}` and
`SPEAK-ASR/youtube-sinhala-asr-preprocessed`. A later experiment (exp-10, see
below) instead used `SPEAK-ASR/openslr-sinhala-asr-norm-noise-rem-preprocessed`
(normalized, noise-removed) -- this repo has no script for that variant; its
exact construction is unverified here. *(Unknown.)*

## Techniques

- LoRA: rank 32, alpha 64, dropout 0.05, targets `q_proj/v_proj/k_proj/o_proj`
  only -- narrower than this project's wide targets (which also include
  `fc1`/`fc2`). *(Verified artifact.)*
- NEFTune noise-embedding regularization (`neftune_noise_alpha=5.0`) --
  not something this project has tried. *(Verified artifact.)*
- Optuna automated hyperparameter search directly minimizing `eval_wer` on
  their own loaded "test" split. *(Verified artifact.)* This is the most
  transferable technique here -- more efficient than manual grid pilots, and
  directly applicable to this project's own already-planned rank/LR ablation.
- `openai/whisper-medium` attempted in later experiments (0.8B params, not
  just whisper-small). *(Verified artifact.)*
- Evaluation: only WER is computed during training
  (`src/components/evaluator.py`); no CER anywhere in the actual training
  loop. A separate, more complete notebook (`notebooks/asr_evaluation.ipynb`)
  adds CER/MER/WIL, but every one of its cells has zero saved output -- it has
  apparently never been executed and committed with real results.
  *(Verified artifact.)*
- Text normalization for scoring uses
  `transformers.models.whisper.english_normalizer.BasicTextNormalizer` --
  Whisper's **English** normalizer, applied to Sinhala text. Not designed for
  Sinhala orthography, compounds, or punctuation conventions.
  *(Verified artifact.)*
- No strict-vs-canonical metric distinction, no confidence intervals, no
  per-subgroup (speaker/domain/duration) breakdown, no error taxonomy --
  substantially thinner evaluation than this project's protocol.
  *(Interpretation, from reading the full evaluator and notebook.)*

## Results (from HF model cards -- nothing is reproducible from this repo)

No metrics are committed anywhere in this git repository. Every number below
comes from the auto-generated Trainer model-card table on the model's public
HF Hub page, fetched directly. These are self-reported training-time WER on
whatever split their pipeline loaded as `"test"` -- not independently computed
here, not checked against any external reference.

| Model | Base | Data | Steps/epochs | Final WER |
|---|---|---|---|---:|
| `whisper-si-exp-1` | small | 1/3 OpenSLR | 2000 steps / 3 ep | not logged (val loss 0.2186 only) |
| `whisper-si-exp-5` | small | full combined (OpenSLR 1-3 + YouTube) | 21,000 steps / 5 ep | 21.60% |
| `whisper-si-exp-6` | small | same as exp-5 | 16,500+ steps / 15 ep | 21.24% |
| `whisper-si-exp-10-medium-all` | **medium** | `...norm-noise-rem-preprocessed` + YouTube, Optuna-tuned LR | 20 ep | **10.85%** |
| `whisper-medium-si-merged` | medium | undocumented | undocumented | undocumented (blank model card) |

*(Recorded result for every row -- fetched from HF, not reproduced.)* Note
exp-5 to exp-6 spent 3x the epochs for essentially the same final WER
(21.60% -> 21.24%) -- a plateau signature similar in shape to this project's
own diminishing-returns data-scale curve, on a completely different axis
(epochs over a fixed dataset, not dataset size).

## Can their metrics be verified?

**No, not from what is committed.** The evaluation notebook that would
independently score a model against a real benchmark has never been run and
saved. Every number is the training pipeline's own self-reported eval-split
WER, sourced from an HF model card auto-generated by the same `Trainer` that
produced the checkpoint -- not an independent check.

## Is the approach rational? The central problem

**The split methodology is confirmed broken, the same way as the historical
`Yohan_Finetune` work, independently.**
`dataset_scripts/openslr-dataset-scripts/load_split_push.py`:

```python
train_test = dataset["train"].train_test_split(test_size=0.3, seed=42)
val_test = train_test["test"].train_test_split(test_size=0.67, seed=42)
```

A plain i.i.d. random split with **zero speaker or session grouping**, on a
source dataset that carries no speaker column at all. Checked every branch's
diff against `main` (`experiment-6` through `experiment-9`, `ds-prep`,
`new-pipeline`, `dev`, `hyper-pram-tuning`): none touch split methodology --
the changes are streaming support, LoRA/training hyperparameters, Optuna, and
deployment scripts (`vast_startup.sh`). The split is never revisited.
*(Verified artifact.)*

Using the same arithmetic as the historical-audit confirmation: this
project's own v4 manifest independently measures 471 speakers across the same
~185,000-row corpus family (~393 rows/speaker average). A random 30%-to-holdout
split on that ratio makes it a near mathematical certainty that most
speakers with any meaningful number of recordings appear in both the training
data and the "test" split their WER is measured against. *(Interpretation,
same reasoning already applied and confirmed for the historical checkpoint.)*

The Optuna hyperparameter search **compounds** this: its objective is
`eval_wer` computed on that same leaky split, so even the "best" tuned
hyperparameters were selected for whatever the leak rewards, not necessarily
for genuine generalization to unseen speakers.

**Conclusion: none of these WER numbers, including the eye-catching 10.85% on
whisper-medium, should be treated as evidence of real generalization
performance until re-measured on a genuinely speaker-disjoint benchmark** --
this project's own frozen, audio-verified, zero-speaker-overlap v4 validation
set is exactly such a benchmark, and none of these checkpoints have been
evaluated against it here.

## What this project can actually learn or gain

- **Adopt**: Optuna-style automated hyperparameter search for the already-
  planned LoRA rank/LR ablation step -- more efficient than manual grid
  pilots, and this codebase has a working reference implementation to learn
  from (not to copy uncritically, since it was tuning against a leaky
  objective).
- **Consider**: NEFTune noise embeddings as a cheap addition to try alongside
  the ablation.
- **Reinforced, not new, lesson**: this is now the *second* independently
  reviewed codebase in the same project ecosystem with the identical
  structural flaw -- a random split with no speaker awareness on a corpus
  where very few speakers each contribute very many rows. That pattern, seen
  twice, is a strong argument for treating *any* previously reported Sinhala
  WER from this ecosystem as unverified until its split is checked, not a
  one-off mistake.
- **Possible re-evaluation candidate**: if any of these checkpoints (in
  particular `whisper-si-exp-10-medium-all`, the medium-model result) can be
  downloaded, they are worth running through this project's own frozen,
  hash-verified, speaker-disjoint v4 validation set purely as a comparison
  point -- same treatment already planned for the historical 17% checkpoint.
  Their engineering practice (config management, Hub sync, Optuna) is
  genuinely stronger than what existed here before this project's rebuild;
  their evaluation science is not, for the same reason the historical
  Yohan work was untrustworthy.
