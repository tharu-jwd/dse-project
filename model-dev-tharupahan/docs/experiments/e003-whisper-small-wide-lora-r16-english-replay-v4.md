# E003: English Replay Wide-LoRA

## Status

Replay preparation complete; mixed-bundle and remote smoke verification are in
progress. No E003 training result exists yet.

## Question

Can a 10% standalone-English replay mixture retain E002's Sinhala improvement
while preventing its measurable English regression?

## Controlled change

E003 retains E002's official `openai/whisper-small` base, wide rank-16 LoRA
targets, 10,000-row Sinhala training bundle, 500 steps, effective batch size,
learning rate, seed family, validation data and decoding. It adds 1,111 English
rows, making English replay 10.00% of the combined 11,111-row pool. It does not
continue training from E002; both experiments start independently from the same
untouched base model so the comparison isolates the replay treatment.

Existing E002 rows retain their Sinhala decoder prefix. New English replay rows
receive Whisper's English decoder prefix. This per-row prefixing is required:
putting English audio under the Sinhala language token would not faithfully
rehearse standalone English recognition.

## Replay source and leakage controls

Replay comes only from LibriSpeech `train-clean-100`, selected deterministically
and speaker-balanced at repository revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. LibriSpeech is distributed under
CC BY 4.0, and its official corpus separates training from test-clean. E003
never samples from `validation`, `test`, `test-clean`, or the project's frozen
2,620-row English-retention bundle.

Sources:

- [Official OpenSLR LibriSpeech corpus and license](https://www.openslr.org/12)
- [Hugging Face dataset split definition](https://huggingface.co/datasets/openslr/librispeech_asr/blob/main/README.md)

The prepared replay artifact must record the pinned source revision, every
source Parquet hash, selected IDs, audio hashes, row/speaker/hour counts and its
own transport hash. The final mixed bundle must prove that all 10,000 E002 rows
are unchanged and that no English-retention sample ID is present.

Replay preparation completed on a CPU-only Colab runtime. The result contains
1,111 rows from 251 speakers and 3.9025 hours of audio. Remote and local
SHA-256 both equal
`0246f185b9f08a79e1eec91f7effb07e7912dd3199b00d78a72cfef61c54781b`.
Direct verification against the 2,620-row test-clean retention benchmark found
zero overlapping sample IDs and zero overlapping audio hashes.

The mixed shard-manifest fingerprint is
`71de2a4fd6ecec0418b0810fabf000d8e4dd40f4427dca0e1b0216b0b7485368`.
It references the ten unchanged, hash-verified E002 shards plus four replay
shards; it does not duplicate or rewrite E002 audio locally.

Four requests across three consecutive work cycles to allocate the prerequisite
T4 smoke runtime returned HTTP 503 `Service Unavailable` before any session was
created. `colab sessions` confirmed after each cycle that no request left a
ghost allocation, so no GPU time or experiment attempt was consumed.

Kaggle was then added as the free-GPU fallback. Authentication, the official
CLI, and the account quota were verified (30 GPU hours available). The private
datasets `tharupahan/sinhala-asr-e003-inputs` and
`tharupahan/sinhala-asr-e003-runtime` contain the hash-checked training inputs
and an offline, pinned Whisper/Transformers/PEFT runtime respectively. The
private kernel `tharupahan/sinhala-asr-e003-resume-smoke` is configured for a
Tesla T4 and cannot start full training: it runs only the one-step/checkpoint/
resume-to-step-two gate.

Kaggle smoke versions 1-2 exposed unavailable outbound DNS and caused no
training. Version 3 proved the offline dependency installation works, then
exposed an incorrect assumption about Kaggle's dataset mount name. Version 4
included manifest-based mount discovery and version 5 repeated it, but both
were assigned workers on which `torch.cuda.is_available()` was false even
though the server-returned kernel metadata recorded `enable_gpu: true` and
`machine_shape: NvidiaTeslaT4`. The CUDA guard stopped both before their first
optimizer step. Kaggle still reports 0.00 of 30.00 GPU hours used. Logs and
structured failure records are preserved under the E003 attempt directory.
This is currently an accelerator-allocation failure, not a training result.
Do not submit the 500-step run until one bounded smoke run sees CUDA and proves
checkpoint resumption.

Once a T4 is available, `stage_e003_colab.py` locally re-hashes all 14 shards,
creates the isolated remote workspace and uploads the manifest, validation set,
resolved config and exact runner scripts. It deliberately does not launch the
job. After phase A reaches checkpoint 1, `resume_e003_smoke_colab.py` refuses to
continue unless the remote process has exited successfully, then installs the
phase-B config and launches the resume. This replaces manual file-by-file
staging while retaining explicit preflight and phase boundaries.

## Decision rule

The thresholds are frozen before training. E003 must retain at least 75% of
E002's improvement over E001 on both canonical Sinhala metrics: WER at or below
101.69% and CER at or below 52.04%. Its English canonical WER may be at most
0.50 percentage points above untouched Whisper-small, and the upper end of the
paired 95% interval may be at most +1.00 point. Report exact paired confidence
intervals and error operations; do not select it from aggregate WER/CER alone.
