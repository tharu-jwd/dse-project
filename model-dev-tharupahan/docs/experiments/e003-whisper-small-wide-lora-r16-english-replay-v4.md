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

The first two attempts to allocate the prerequisite T4 smoke runtime returned
HTTP 503 `Service Unavailable` before any session was created. `colab sessions`
confirmed that neither request left a ghost allocation, so no GPU time or
experiment attempt was consumed. In accordance with the two-retry infrastructure
limit, preparation stopped without launching training; a later allocation may
retry the unchanged smoke configuration.

## Decision rule

The thresholds are frozen before training. E003 must retain at least 75% of
E002's improvement over E001 on both canonical Sinhala metrics: WER at or below
101.69% and CER at or below 52.04%. Its English canonical WER may be at most
0.50 percentage points above untouched Whisper-small, and the upper end of the
paired 95% interval may be at most +1.00 point. Report exact paired confidence
intervals and error operations; do not select it from aggregate WER/CER alone.
