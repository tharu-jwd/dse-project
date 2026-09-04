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
