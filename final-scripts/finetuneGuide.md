# Whisper Sinhala Fine-Tuning — Final Scripts

Self-contained folder: everything needed to fine-tune, and evaluate, Whisper-small
for Sinhala ASR. Two fine-tuning strategies are provided so you can compare them
on the same data and the same held-out test set:

| Script | What it does | Output |
|---|---|---|
| `finetune_whisper.py` | **Full fine-tune** — every model weight is trained | A full Whisper checkpoint (~1GB+) |
| `finetune_whisper_lora.py` | **LoRA fine-tune** — base model frozen, a small low-rank adapter is trained | A PEFT adapter directory (a few MB) |
| `evaluate_finetuned.py` | Scores your fine-tuned checkpoints/adapters against the test set — pass as many `--model`/`--lora` runs as you want compared, ranked by WER, to find the best one | Ranked console summary + one `<model>_predictions.csv` per model |
| `prepare_whisper_dataset.py` | Shared library both fine-tune scripts import (audio decode + optional noise augmentation + feature extraction + collation). | — |

Both fine-tune scripts share the same CLI shape, the same data pipeline, and
log to the same W&B project, so `--run-name`s from either one show up side by
side for comparison.

## Dataset paths are fixed, not CLI flags

This whole folder is meant to be uploaded to the GPU pod (RunPod or
otherwise) as a unit, **with the data included** — so `finetune_whisper.py`,
`finetune_whisper_lora.py`, and `evaluate_finetuned.py` all read the dataset
from a fixed location next to the scripts, instead of taking a path on the
command line:

```
final-scripts/
  finetune_whisper.py
  finetune_whisper_lora.py
  evaluate_finetuned.py
  prepare_whisper_dataset.py
  requirements.txt
  data/
    stratified/
      train.parquet         <- used by both fine-tune scripts
      validation.parquet    <- used by both fine-tune scripts
      test.parquet          <- used by evaluate_finetuned.py
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
       gs://singen/whisper/finalData/stratified/train.parquet \
       gs://singen/whisper/finalData/stratified/validation.parquet \
       gs://singen/whisper/finalData/stratified/test.parquet \
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

### Audio augmentation

Both scripts apply three independent waveform augmentations to a fraction of
the **training** clips only (the eval/test split is always kept clean, so
WER/CER stays comparable across runs). Each one is applied per-sample with
its own probability, so a clip can get any combination of them (or none).

| Augmentation | Flags | Defaults | Meaning |
|---|---|---|---|
| Noise injection | `--noise-prob`, `--noise-snr-db-min/max` | `0.2`, `20.0`/`30.0` | Mixes in light Gaussian noise at a random SNR (dB) — lower dB = louder/more noise |
| Time stretching | `--stretch-prob`, `--stretch-rate-min/max` | `0.2`, `0.9`/`1.1` | Speeds up/slows down the clip by a random rate (>1 = faster, <1 = slower), to cover varying speaking paces |
| Pitch shifting | `--pitch-prob`, `--pitch-semitones-min/max` | `0.2`, `-2.0`/`2.0` | Shifts pitch by a random number of semitones, to help generalize across voice types (e.g. child vs. adult speakers) |

Set any `--*-prob` to `0` to disable that augmentation. All three are on by
default at light settings, so no extra flags are needed to use them:

```bash
# override strength
python3 finetune_whisper_lora.py ... --noise-prob 0.3 --noise-snr-db-min 15 --noise-snr-db-max 25
python3 finetune_whisper_lora.py ... --stretch-prob 0.3 --pitch-prob 0.3

# disable all three
python3 finetune_whisper_lora.py ... --noise-prob 0 --stretch-prob 0 --pitch-prob 0
```

Noise injection here is synthetic white noise, not real-world background
noise (traffic, babble, etc.) — there's no noise-clip corpus (e.g. MUSAN) in
this project yet. Background-noise overlay was considered but skipped until
such a corpus is added.

### W&B

Both scripts log to Weights & Biases when `--wandb-project` is passed. Log
in once per machine before your first run:

```bash
wandb login
```

Omit `--wandb-project` to disable logging entirely (e.g. for the CPU smoke
test below).

### MLflow experiment tracking (optional)

MLflow tracking is **opt-in** — pass `--use-mlflow` to either script and it
logs params, per-epoch WER/CER, and final metrics into a shared experiment
so the whole team can compare each other's runs in one place. Leave it off
(the default) and neither script touches MLflow at all: no import, no
network call, no need to even have `mlflow` installed or a server running.
Use it if your team wants shared tracking; skip it if you're just running a
one-off locally or prefer W&B alone.

```bash
pip install mlflow>=2.14.0,<3.0.0   # only needed if you're using --use-mlflow

python3 finetune_whisper_lora.py \
    --use-mlflow \
    --output-dir /workspace/whisper-small-sinhala-lora/run1 \
    --run-name run1-lr1e-4-bs32 \
    --learning-rate 1e-4 --per-device-train-batch-size 32 --num-train-epochs 4
```

**No account needed anywhere.** MLflow here is self-hosted, not a SaaS
product — there's a tracking server running as a Docker Compose service
(`mlflow` in the repo-root [`docker-compose.yml`](../docker-compose.yml)),
and "logging in" just means pointing your script at that server's URL. As
currently set up there's no username/password on the server itself — anyone
who can reach the port can read and write runs. That's fine for a small
trusted team on a private network/VPN; if the server's port ever needs to be
open to the wider internet (likely, since fine-tuning runs happen on RunPod
pods, not on whoever's machine runs `docker compose up`), put it behind a
firewall allowlist, an SSH tunnel, or a reverse proxy with basic auth rather
than leaving `MLFLOW_PORT` open to everyone.

**One-time setup (whoever runs the shared server):**
```bash
# from the repo root, alongside docker-compose.yml
docker compose up -d mlflow
```
This starts the tracking server on `MLFLOW_PORT` (default `5000`, see
[`.env.example`](../.env.example)), backed by a Docker volume (`mlflow_db`
for run/param/metric metadata, `mlflow_artifacts` for any uploaded
checkpoints) so history survives container restarts.

**Every teammate / every GPU pod**, before running a fine-tune, points at
that server once per machine:
```bash
export MLFLOW_TRACKING_URI=http://<host-running-docker-compose>:5000
```
(On a RunPod pod this needs the host's public IP/hostname and the port to
be reachable from the pod — an SSH tunnel is the simplest way to do this
without opening the port publicly: `ssh -N -L 5000:localhost:5000 you@host`
run from the pod, then `MLFLOW_TRACKING_URI=http://localhost:5000`.)

If `MLFLOW_TRACKING_URI` is unset, the scripts still work — MLflow just
falls back to a local `./mlruns` folder, tracked but not shared with anyone.

**Checking for already-tried hyperparameters:** before training starts,
both scripts search the MLflow experiment for a previous run with the exact
same hyperparameters (learning rate, batch size, epochs, augmentation
settings, LoRA config, etc.) and print a warning listing any matches (run
name, WER, run ID) if found. It's a heads-up, not a hard stop — a
deliberate re-run is still allowed — but it means you'll see it before
spending GPU hours repeating a run a teammate already logged.

**Viewing runs:** open `http://<host>:5000` in a browser (same tracking
server; MLflow's UI is served from the same port) to compare runs side by
side, sort by WER, and inspect params/metrics/artifacts per run.

Flags:
```bash
--use-mlflow                             # turn MLflow tracking on for this run (off by default)
--mlflow-experiment my-experiment-name   # default: whisper-sinhala-finetune; only used with --use-mlflow
--mlflow-log-artifacts                   # also upload the saved checkpoint/adapter
                                          # to MLflow (off by default: full
                                          # checkpoints are ~1GB+; adapters are small
                                          # and cheap to include); only used with --use-mlflow
```

### Smoke test (no GPU, before committing to a real run)

Both scripts accept `--smoke-test`: 2 training steps, no generation-based
eval, runs on CPU. Still reads real rows from `data/stratified/`, so make
sure §2 is done first, even for this:

```bash
python3 finetune_whisper.py --smoke-test
python3 finetune_whisper_lora.py --smoke-test
```

## 4. Evaluating on the test set — finding your best model

`evaluate_finetuned.py` reads `data/stratified/test.parquet` and scores
whichever of your own fine-tuned runs you point it at: a full checkpoint via
`--model` (repeatable) and/or a LoRA adapter via `--lora <adapter-dir>:<base-model>`
(repeatable). Pass every run you want to compare in one invocation — it
prints a summary **ranked by WER, best first**, and tells you the winner.

```bash
# compare a full fine-tune against a LoRA run
python3 evaluate_finetuned.py \
    --model /workspace/whisper-small-sinhala/run1-lr3e-5-bs32 \
    --lora  /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32:openai/whisper-small

# compare several checkpoints from a hyperparameter sweep to pick the best
python3 evaluate_finetuned.py \
    --model /workspace/whisper-small-sinhala/run1-lr1e-5-bs16 \
    --model /workspace/whisper-small-sinhala/run2-lr3e-5-bs16 \
    --model /workspace/whisper-small-sinhala/run3-lr3e-5-bs32

# plus log the ranked summary to the same W&B project as training
python3 evaluate_finetuned.py \
    --model /workspace/whisper-small-sinhala/run1-lr3e-5-bs32 \
    --lora  /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32:openai/whisper-small \
    --wandb-project whisper --run-name model-selection
```

Output: a console table sorted by WER (top row = best model), a `Best
model: ...` line, and one `<model-name>_predictions.csv` per model
(reference vs. prediction, for error inspection) under `--output-dir`
(defaults to the current directory).

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
7. Run `evaluate_finetuned.py` (§4) against every run you want to compare —
   it ranks them by WER so you can pick the best one to report/ship.
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

## 7. MLflow — step-by-step summary (RunPod + GCP dataset setup)

End-to-end checklist for tracking a fine-tune in MLflow when training runs
on a RunPod GPU pod, the dataset lives in a GCP bucket, and the MLflow
server runs on an always-on host (not your laptop — see §"MLflow experiment
tracking" above for why).

1. **Start the MLflow server, once, on the always-on host** (not RunPod, not
   a laptop you'll close):
   ```bash
   cd dse-project
   docker compose up -d mlflow
   ```
2. **Open its port to the machines that need it** — your team's IPs and
   RunPod's egress — in that host's firewall/security group. `MLFLOW_PORT`
   defaults to `5000` (see `.env.example`).
3. **Launch the RunPod GPU pod** (A100/A10 template with CUDA preinstalled).
4. **Get `final-scripts/` onto the pod** — zip + `runpodctl send`/`receive`,
   or `git clone` (§2 Method B / §5 steps 2–3).
5. **Install dependencies on the pod** (MLflow tracking is opt-in, so also
   install the `mlflow` client since you're using it here):
   ```bash
   cd final-scripts
   pip install -r requirements.txt
   pip install "mlflow>=2.14.0,<3.0.0"
   ```
6. **Pull the dataset from the GCP bucket onto the pod:**
   ```bash
   gcloud auth activate-service-account --key-file=/workspace/gcp-key.json
   mkdir -p data/stratified
   gsutil -m cp \
       gs://singen/whisper/finalData/stratified/train.parquet \
       gs://singen/whisper/finalData/stratified/validation.parquet \
       gs://singen/whisper/finalData/stratified/test.parquet \
       data/stratified/
   ```
7. **Point the pod at the MLflow server:**
   ```bash
   export MLFLOW_TRACKING_URI=http://<always-on-host-ip-or-domain>:5000
   ```
8. **Run the fine-tune with `--use-mlflow`** (tracking is opt-in — omit the
   flag and none of this applies, no server or package needed):
   ```bash
   python3 finetune_whisper_lora.py \
       --use-mlflow \
       --output-dir /workspace/whisper-small-sinhala-lora/run1 \
       --run-name run1-lr1e-4-bs32 \
       --learning-rate 1e-4 \
       --per-device-train-batch-size 32 \
       --num-train-epochs 4
   ```
   Confirm the connection from the printed line right after startup:
   `MLflow tracking: http://<host>:5000  experiment=whisper-sinhala-finetune`.
   If the same hyperparameters were already run by a teammate, a warning
   listing that run (name, WER, run ID) prints here too — not a blocker,
   just a heads-up.
9. **Watch it live** — from any machine, open
   `http://<always-on-host-ip-or-domain>:5000`, click the
   `whisper-sinhala-finetune` experiment, then the run name. Params show up
   immediately; WER/CER metric charts fill in as each epoch finishes.
10. **Compare runs** once you have more than one: select several runs on the
    experiment page → **Compare**, for a side-by-side param/metric table.
11. **(Optional) Keep the checkpoint in MLflow too:** add
    `--mlflow-log-artifacts` to upload the saved checkpoint/adapter as a run
    artifact (off by default — full checkpoints are ~1GB+).
12. **After training**, copy the result off the pod before terminating it
    (pod disk is ephemeral) — `runpodctl send`/`receive` as in §5 step 8.
    The MLflow run itself doesn't need copying — it already lives on the
    always-on server, independent of the pod.
