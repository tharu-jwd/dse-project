# Colab CLI Operations and Storage Safety

## Default policy from E002 onward

Google Drive is not mounted. Colab is treated as disposable compute while the
local project is the durable source of truth:

```text
local reports/colab inputs
  -> /content/sinhala-asr-job/input on a named Colab session
  -> bounded training/evaluation
  -> /content/sinhala-asr-job/output
  -> local reports/experiments/eNNN-...
  -> checksum and completeness verification
  -> terminate the Colab session
```

This prevents training code from reading, modifying, or deleting Google Drive
content. Inputs and outputs must stay below `/content/sinhala-asr-job`; job
scripts must not use recursive deletion or unresolved paths. The test split is
uploaded only for the final locked evaluation.

Every job has a fixed step/time bound. For longer training, download a complete
checkpoint and trainer state after each stage, verify them locally, and upload
that checkpoint to the next disposable session. A Colab runtime must not be
stopped until required outputs are present locally and their hashes are
recorded. Unexpected free-tier termination can lose only the current bounded
stage, never an earlier verified checkpoint.

## Failure containment and recovery

Training runs as a supervised background process rather than one long blocking
CLI request. Every experiment has immutable attempt directories:

```text
reports/experiments/eNNN-.../
  attempts/attempt-001/
    events.jsonl
    remote-stdout.log
    session.json
    resolved-config.json
    failure.json              # only after a failure
  checkpoints/checkpoint-N/
  latest-verified.json
```

Before training, the operator verifies input hashes, available disk, GPU type,
package versions, adapter/base parameter counts, and a two-step smoke run. A
failed preflight does not begin the measured experiment.

During training:

1. Write structured progress and heartbeat events at least every five steps.
2. Save adapter, optimizer, scheduler, trainer, scaler, and RNG state every 25
   steps for short pilots. At E001 throughput this limits uncheckpointed work
   to roughly two minutes.
3. Write a `COMPLETE` marker only after every checkpoint file is closed.
4. Package only a checkpoint bearing that marker, calculate its SHA-256 on the
   runtime, and download it immediately to a local `.partial` path.
5. Recalculate the hash locally, verify required files and recorded step, then
   atomically rename it into `checkpoints/checkpoint-N/` and update
   `latest-verified.json`.
6. Poll session state, process state, disk use, heartbeat age, and logs. Never
   treat a quiet CLI call as proof that training is healthy.

Failure handling is explicit:

- **Training process exits:** download logs and any completed checkpoint, write
  `failure.json` with exit code and last verified step, then stop the runtime.
- **CLI transport fails but runtime is alive:** reconnect and inspect status;
  do not launch a duplicate process. Retry transport operations with bounded
  backoff.
- **Runtime disappears:** record the lost step interval, allocate a fresh
  session, upload the latest locally verified checkpoint, and resume with the
  same optimizer, scheduler, RNG state, dataset order, and resolved config.
- **GPU unavailable/quota exhausted:** leave the experiment pending. Do not
  repeatedly allocate sessions or change hardware silently.
- **OOM:** preserve evidence and stop. Batch size may change only in a new
  documented attempt, with gradient accumulation adjusted to preserve effective
  batch size; it is not an automatic hidden retry.
- **Bad sample/data exception:** record the sample ID and stop. Never skip it
  silently; any exclusion requires a new auditable dataset decision.
- **NaN/divergence:** retain the preceding checkpoint and metrics. Hyperparameter
  changes require a new attempt/configuration rather than overwriting the run.

Only infrastructure/transport failures may auto-resume, with at most two
automatic attempts. Model, data, OOM, and numerical failures require diagnosis
first. Resumption is verified by checking that the first reported global step
is the saved step and that optimizer/scheduler state loaded successfully.

After success, predictions, metrics, final adapter, logs, resolved config, and
checkpoint hashes are downloaded and verified. Session termination occurs in a
final cleanup step after verification; on any exception the operator captures
available evidence before issuing `colab stop`.

## CLI setup

The local operator uses `google-colab-cli==0.6.0`. That release is incompatible
with `jupyter-kernel-client>=1`; install the known-compatible dependency:

```bash
uv tool install --force \
  --with 'jupyter-kernel-client==0.9.0' \
  google-colab-cli
```

CLI OAuth is completed by the owner directly in their terminal. Authorization
codes and tokens are never copied into chat or committed. Session lifecycle:

```bash
colab new -s JOB_ID --gpu T4
colab upload -s JOB_ID LOCAL_INPUT /content/sinhala-asr-job/input/FILE
colab exec -s JOB_ID -f LOCAL_SCRIPT --timeout SECONDS
colab download -s JOB_ID /content/sinhala-asr-job/output/FILE LOCAL_OUTPUT
colab stop -s JOB_ID
```

Free-tier GPU availability and runtime lifetime are not guaranteed. Always
inspect `colab status`, use measured throughput, and keep stages short.

## Historical exception

E001 was run manually before this policy and saved its adapter under
`MyDrive/sinhala-asr/e001-whisper-small-wide-lora-r16-100-step-v4`. That path is
part of the experiment record. It does not authorize Drive mounting for E002 or
later jobs.
