# Historical Training Audit

This document is an evidence-graded guide to the work inherited from the team.
It is not a declaration that every historical note is correct. The branch
history, committed code, prediction exports, and reports were checked against
one another. Claims that still depend on an external tracker or an unavailable
checkpoint are explicitly marked as unverified.

## Scope and evidence rules

The named Yohan branches currently have no file differences from `main`; their
changes were merged. Their commit histories remain useful for reconstructing
the sequence of work:

- `Yohan_Observation`: data preparation and exploratory notebooks
- `Yohan_Finetune`: full fine-tuning and LoRA scripts
- `Yohan_augmentation`: noise, speed, pitch, scheduling, normalization,
  tracking, dependency fixes, and English-forgetting evaluation
- `Yohan_RealTime`: real-time inference work

Evidence labels used below:

- **Verified artifact**: reproducible from committed code or prediction files
- **Recorded result**: present in committed tracking/results, but the original
  run and checkpoint were not independently reproduced
- **Interpretation**: conclusion drawn from the verified artifacts
- **Unknown**: evidence is absent or inaccessible

The authoritative data location supplied by the project owner is
`gs://singen/whisper/finalData/`. It has not yet been inventoried because this
environment is authenticated but Google Cloud rejects bucket access because the
owning project's billing account is delinquent. The local historical data and
results therefore cannot yet be proven identical to the bucket. Independently
downloadable upstream sources will be audited first; bucket files will later be
treated as team-processed comparison artifacts.

## Run inventory

| Historical run | Setup | Sinhala test WER/CER | English WER/CER | Evidence and limitations |
|---|---|---:|---:|---|
| Full fine-tune | Whisper-small, LR `3e-5`, linear schedule, 4 epochs, effective batch 32 | 17.08/3.49 recorded; 17.36 WER recomputed | 80.86/72.05 | Prediction CSV verifies 17.36 WER under the current normalizer. Tracker values and hyperparameters are recorded results. Severe English forgetting. |
| Narrow LoRA | q/v targets, LR `5e-5`, linear schedule, effective batch 64 | 21.01/5.73 | 73.74 WER | Recorded result. The run name reportedly said `3e-5`, while code/tracking records `5e-5`. |
| Interrupted LoRA | LR `3e-5`, cosine schedule | 59.07/18.03 | Unknown | Not a valid final comparison because training was interrupted. |
| Wide LoRA | q/k/v/out/fc1/fc2 targets, LR `1e-4`, cosine schedule | 25.99/7.06 | 7.03 WER | Sinhala prediction CSV verifies 25.99 WER. It completed after interruptions/resumes; causal attribution to target width is not possible because several factors changed. |
| Later full fine-tune | LR `1e-5`, cosine schedule, effective batch 64, `stratified_v2` | 28.47/7.93 | 6.06 WER | Recorded result. Not directly comparable with runs using another dataset version. |

`CER < 10%` was already achieved by several historical runs. The unmet headline
target is `WER < 10%`, while retaining an honest, fixed evaluation protocol.

## What the prediction artifacts establish

The best historical full fine-tune has 15,483 exported predictions:

- WER recomputed with the repository normalizer: 17.36%
- raw exact match: 53.77%
- normalized exact match: 60.42%
- whitespace-insensitive exact match: 76.54%
- Sinhala-only: 15,258 samples, 16.58% WER
- code-switched: 225 samples, 29.77% WER
- no empty predictions or detected repetition loops
- WER rises from 13.90% for 1–2 word samples to 35.04% for 21+ words

The wide-LoRA export has 25.99% WER, 34.20% raw exact match, 42.45%
normalized exact match, 50.58% whitespace-insensitive exact match, and three
detected repetition loops. Its Sinhala-only/code-switched WER is 24.77%/45.50%.

These figures show that spacing and orthographic policy account for a material
share of counted errors, but not all of them. Genuine acoustic substitutions,
deletions, names, numbers, code-switching, and long-utterance failures remain.
Both strict and canonical metrics must therefore be retained.

The old prediction CSVs omit stable sample IDs, audio hashes, source, speaker,
and duration. They cannot support source-, speaker-, or duplicate-aware error
analysis. This is a limitation of the artifacts, not evidence that those
subgroups are healthy.

## Data and evaluation risks

The committed documentation describes approximately 149,926 OpenSLR rows and
4,902 collection rows, with an 80/10/10 split. The split code stratifies by
source dataset, not by speaker. Until upstream data is audited and the team's
cloud artifacts become available for comparison, the following remain unknown:

- whether any speaker or audio content crosses splits
- whether the bucket exactly matches the committed split files
- which transcript corrections and exclusions are authoritative
- whether all 268 reportedly flagged collection samples were adjudicated
- whether historical checkpoints were trained on exactly the evaluated data

No new headline result should be accepted until the dataset and split
fingerprints are recorded and leakage checks pass.

## English and code-switch evaluation policy

English words inside Sinhala utterances must not be silently removed from the
main references. They are part of the transcription task; deleting them would
alter alignment and make WER incomparable and artificially easier.

Every candidate will report:

1. Overall strict and canonical WER/CER, including code-switched words.
2. Sinhala-only utterance WER/CER.
3. Code-switched utterance WER/CER.
4. Sinhala-token and English-token error rates within code-switched utterances,
   once a reviewed token-language annotation is available.
5. Standalone English retention on a fixed English test set.

English preservation and Sinhala accuracy are not inherently mutually
exclusive. The historical results demonstrate that the training recipe
matters: full and narrow-LoRA runs forgot English badly, while a wide-LoRA run
retained far more English but had worse Sinhala WER. Because learning rate,
scheduler, adapter targets, data version, and interruptions were confounded,
these results do not prove that one adaptation method caused either outcome.

## Experiment decisions

All experiments use the same frozen split, text policy, decoding protocol, and
evaluation suite.

### E0 — untouched baseline

Evaluate official `openai/whisper-small` without training. This establishes the
true starting point and validates the evaluation pipeline.

### E1 — clean Sinhala baseline

Fine-tune the official checkpoint with a conservative, pilot-selected learning
rate. This is the reproducible primary line of work.

### E2 — bilingual replay

Repeat the winning E1 recipe while mixing a small, fixed English rehearsal set
and the available Sinhala-English code-switched samples. Tune the mixing weight
on validation results. This directly tests whether Sinhala gains and English
retention can coexist.

### E3 — parameter-efficient adaptation

Run wide-target LoRA or DoRA with the same data and evaluation as E1/E2. Treat
this as a compute/retention comparison, not an assumed improvement.

### E4 — continue the historical 17% checkpoint

Only if the exact checkpoint, optimizer state, dataset identity, and license
are recovered, continue it with a low learning rate as a separate
Sinhala-specialization experiment. It may be a cheap path to lower Sinhala WER,
but it is not a clean baseline and starts from severe English forgetting.

At most two pilot winners advance. A larger model is out of scope until a
Whisper-small recipe has passed the target-oriented gates.

## Operational lessons retained from the old work

- Validate dependency versions and `accelerate` before renting a GPU.
- Check tokenizer truncation and long-utterance behavior explicitly.
- Save resolved configs and actual learning rates; names alone were misleading.
- Test interruption/resume before a full run.
- Use normalization consistently, but never use it to conceal model errors.
- Do not infer causality from runs that changed data, LR, schedule, batch size,
  adapter targets, or interruption history simultaneously.
- Export row-level predictions with full manifest metadata for every run.
- Record elapsed GPU time and actual cost; historical records do not contain
  enough information to reconstruct cost efficiency.

## Current verdict

The historical work is useful as evidence and as a source of failure cases, but
not as a trustworthy end-to-end training system. The 17% model is a legitimate
comparison checkpoint and possible continuation candidate. It does not replace
a clean baseline, and its standalone-English regression makes it unsuitable as
the only starting point.

The next evidence task is downloading and fingerprinting the independently
available upstream datasets, then running the manifest audit. Restored GCS
billing remains necessary to compare the team's processed files and recover any
verified corrections, but it does not block the upstream audit.
