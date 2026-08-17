# Whisper Sinhala Fine-Tuning — Final Scripts

Self-contained folder: everything needed to fine-tune, and evaluate, Whisper-small
for Sinhala ASR. Two fine-tuning strategies are provided so you can compare them
on the same data and the same held-out test set:

| Script | What it does | Output |
|---|---|---|
| `finetune_whisper.py` | **Full fine-tune** — every model weight is trained | A full Whisper checkpoint (~1GB+) |
| `finetune_whisper_lora.py` | **LoRA fine-tune** — base model frozen, a small low-rank adapter is trained | A PEFT adapter directory (a few MB) |
| `evaluate_baselines.py` | Scores any model (published baseline, your full fine-tune, or your LoRA adapter) on the test set: WER/CER | Console summary + `baseline_predictions.csv` |
| `prepare_whisper_dataset.py` | Shared library both fine-tune scripts import (audio decode + feature extraction + collation). | — |

Both fine-tune scripts share the same CLI shape, the same data pipeline, and
log to the same W&B project, so `--run-name`s from either one show up side by
side for comparison.

## Dataset paths are fixed, not CLI flags

This whole folder is meant to be uploaded to the GPU pod (RunPod or
otherwise) as a unit, **with the data included** — so `finetune_whisper.py`,
`finetune_whisper_lora.py`, and `evaluate_baselines.py` all read the dataset
from a fixed location next to the scripts, instead of taking a path on the
command line:

```
final-scripts/
  finetune_whisper.py
  finetune_whisper_lora.py
  evaluate_baselines.py
  prepare_whisper_dataset.py
  requirements.txt
  data/
    stratified/
      train.parquet         <- used by both fine-tune scripts
      validation.parquet    <- used by both fine-tune scripts
      test.parquet          <- used by evaluate_baselines.py
```

Nothing else needs to change between runs — just make sure `data/stratified/`
is populated (see below) before running any script, and run everything from
inside `final-scripts/`.

## 1. Install

```bash
cd final-scripts
pip install -r requirements.txt
```

Pinned versions (`torch`, `transformers>=4.41.0,<5.0.0`, `accelerate`, `peft`,
`evaluate`, `jiwer`, `pyarrow`, `soundfile`, `librosa`, `wandb`, `numpy<2.0.0`,
...) are chosen to be mutually compatible — install all of them from this one
file rather than picking versions ad hoc, since `numpy>=2` in particular can
silently break `pyarrow`/`soundfile`/`librosa` at import or ABI level.

## 2. Getting the data into `data/stratified/` — two methods

The training/eval data lives in a GCP bucket. Pick **one** method to
populate `data/stratified/` before training; the scripts themselves don't
care which one you used.

### Method A — Download straight onto the GPU pod from GCS

Do this once, right after the pod comes up and before starting training.

1. Install/confirm `gsutil` is available on the pod (RunPod images usually
   have `gcloud`/`gsutil` preinstalled; otherwise `pip install gsutil` or
   follow the [gcloud install docs](https://cloud.google.com/sdk/docs/install)).
2. Authenticate:
   ```bash
   gcloud auth activate-service-account --key-file=/workspace/gcp-key.json
   # or, interactively: gcloud auth login
   ```
3. Pull the three split files into place:
   ```bash
   mkdir -p final-scripts/data/stratified
   gsutil -m cp \
       gs://<your-bucket>/final_split_dataset/stratified/train.parquet \
       gs://<your-bucket>/final_split_dataset/stratified/validation.parquet \
       gs://<your-bucket>/final_split_dataset/stratified/test.parquet \
       final-scripts/data/stratified/
   ```

### Method B — Zip locally and transfer with `runpodctl send`/`receive`

Useful if you've already downloaded the split locally (e.g. for earlier
experiments) and don't want to hit GCS again, or if the pod has a slow/
unreliable connection to GCS but you have a fast link to your own machine.
This is a direct, peer-to-peer transfer (no intermediate cloud storage, no
SSH keys/IP needed) — this is the method actually used for this project.

1. Make sure `final-scripts/` locally already has the three parquet files
   under `data/stratified/` (pull them from GCS once with `gsutil cp`, same
   paths as Method A, if you don't already have them).
2. Zip the whole folder (scripts + `requirements.txt` + `data/`) on your
   local machine:
   ```bash
   zip -r final-scripts.zip final-scripts/
   ```
3. Install `runpodctl` locally if you don't have it yet (see
   [runpod.io/console/user/settings](https://www.runpod.io/console/user/settings)
   → API Keys, or the [runpodctl releases page](https://github.com/runpod/runpodctl/releases)).
   Send the zip:
   ```bash
   runpodctl send final-scripts.zip
   ```
   This prints a one-time transfer code, e.g. `8544-galileo-tango-alpha-3`.
4. On the RunPod GPU pod's terminal (via the web terminal or SSH), install
   `runpodctl` if it isn't already on the image, then receive:
   ```bash
   runpodctl receive 8544-galileo-tango-alpha-3
   ```
   This downloads `final-scripts.zip` directly into the pod's current
   directory (typically `/workspace`).
5. Unzip it on the pod:
   ```bash
   cd /workspace
   unzip final-scripts.zip
   cd final-scripts
   ```

Either method (A or B) ends the same way: `final-scripts/data/stratified/{train,validation,test}.parquet`
exist on the pod before you run anything.

## 3. Fine-tuning

Run these from inside `final-scripts/`, with `data/stratified/` already
populated (§2).

### Full fine-tune

```bash
python3 finetune_whisper.py \
    --output-dir /workspace/whisper-small-sinhala/run1-lr3e-5-bs32 \
    --run-name run1-lr3e-5-bs32 \
    --wandb-project whisper \
    --learning-rate 3e-5 \
    --per-device-train-batch-size 32 \
    --num-train-epochs 4
```

### LoRA fine-tune

Same flags, plus optional `--lora-r` / `--lora-alpha` / `--lora-dropout` /
`--lora-target-modules` (defaults: `r=32, alpha=64, dropout=0.05,
target_modules=q_proj,v_proj`). LoRA typically wants a higher learning rate
than a full fine-tune (default here is `1e-4` vs `1e-5`):

```bash
python3 finetune_whisper_lora.py \
    --output-dir /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32 \
    --run-name run1-lr1e-4-bs32 \
    --wandb-project whisper \
    --learning-rate 1e-4 \
    --per-device-train-batch-size 32 \
    --num-train-epochs 4
```

### W&B

Both scripts log to Weights & Biases when `--wandb-project` is passed. Log
in once per machine before your first run:

```bash
wandb login
```

Omit `--wandb-project` to disable logging entirely (e.g. for the CPU smoke
test below).

### Smoke test (no GPU, before committing to a real run)

Both scripts accept `--smoke-test`: 2 training steps, no generation-based
eval, runs on CPU. Still reads real rows from `data/stratified/`, so make
sure §2 is done first, even for this:

```bash
python3 finetune_whisper.py --smoke-test
python3 finetune_whisper_lora.py --smoke-test
```

## 4. Evaluating on the test set

`evaluate_baselines.py` reads `data/stratified/test.parquet` and scores any
mix of: published baseline models, your own full fine-tune
(`--custom-model`), and your own LoRA adapter (`--custom-lora`) — all with
the same WER/CER normalization, so results are directly comparable.

```bash
# published baselines only
python3 evaluate_baselines.py

# add your full fine-tune
python3 evaluate_baselines.py \
    --models openai/whisper-small \
    --custom-model /workspace/whisper-small-sinhala/run1-lr3e-5-bs32

# add your LoRA adapter (format: <adapter-dir>:<base-model>)
python3 evaluate_baselines.py \
    --models openai/whisper-small \
    --custom-lora /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32:openai/whisper-small

# both, plus log the summary table to the same W&B project as training
python3 evaluate_baselines.py \
    --models openai/whisper-small \
    --custom-model /workspace/whisper-small-sinhala/run1-lr3e-5-bs32 \
    --custom-lora /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32:openai/whisper-small \
    --wandb-project whisper --run-name run1-eval

python3 evaluate_baselines.py --list-models   # see all known baseline repo IDs (no data/ needed)
```

Add `--max-samples 200` for a quick sanity pass before committing to a full
test-set run; omit it for the real number to report.

## 5. Running on RunPod — end to end

1. Launch a GPU pod (an A100/A10 template with CUDA preinstalled is
   simplest — SpecAugment aside, the scripts auto-detect `bf16` on Ampere+
   and fall back to `fp16` otherwise, no manual flag needed).
2. On your local machine: zip `final-scripts/` (with `data/stratified/`
   already populated from GCS) and send it with `runpodctl` (§2 Method B):
   ```bash
   zip -r final-scripts.zip final-scripts/
   runpodctl send final-scripts.zip
   ```
   Note the transfer code it prints.
3. On the pod's terminal, receive and unzip:
   ```bash
   cd /workspace
   runpodctl receive <the-code-from-step-2>
   unzip final-scripts.zip
   cd final-scripts
   ```
   (Alternatively, skip steps 2–3 and pull the data straight onto the pod
   with §2 Method A instead.)
4. `pip install -r requirements.txt`
5. `wandb login` if you want experiment tracking.
6. Run `finetune_whisper.py` and/or `finetune_whisper_lora.py` as shown in
   §3. Checkpoints/adapters land under `--output-dir`.
7. Run `evaluate_baselines.py` (§4) to get the final WER/CER for whichever
   run(s) you want to report.
8. Copy the result off the pod before terminating it — pod-local disk is
   ephemeral. `runpodctl send` works in reverse too, from the pod back to
   your local machine:
   ```bash
   # on the pod
   runpodctl send /workspace/whisper-small-sinhala/run1-lr3e-5-bs32
   # on your local machine, with the printed code
   runpodctl receive <the-code>
   ```

## 6. Running locally

Only realistic for a smoke test or a small-scale run if you have a capable
GPU — the full `stratified/train.parquet` split is large (123,862 rows).

1. `pip install -r requirements.txt` in a virtualenv.
2. Populate `data/stratified/` via §2 Method A or B (Method A — a one-time
   `gsutil cp` — is usually simplest locally).
3. Run the same commands as §3/§4. If you don't have a CUDA GPU locally, use
   `--smoke-test` to validate the pipeline, then move the real run to RunPod.
