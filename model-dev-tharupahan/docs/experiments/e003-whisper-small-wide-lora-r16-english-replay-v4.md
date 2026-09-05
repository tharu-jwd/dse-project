# E003: English Replay Wide-LoRA

## Status

Complete and rejected as a release candidate. E003 improved Sinhala over E002,
but raw-reference English replay made canonical English retention substantially
worse and failed the frozen retention gate.

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
optimizer step. Logs and structured failure records are preserved under the
E003 attempt directory. These were infrastructure diagnostics, not training
results.

After the account owner completed Kaggle phone verification, smoke version 6
received a real CUDA worker and reached the first model forward pass. It exposed
a runner defect: the model was loaded as FP16 while the training feature tensor
arrived as FP32. Version 7 showed that casting inside Transformers'
`BatchFeature` was insufficient because Trainer reconstructed that custom
container. The final correction returns a plain tensor dictionary and explicitly
casts `input_features` to FP16.

Kaggle smoke version 8 then passed the complete gate. Phase A created checkpoint
1 with SHA-256
`698095f85a388038d91ec4da2be54d95c8067f0ba90ebfdf11273c8caab9edc7`;
a fresh process resumed it and created checkpoint 2 with SHA-256
`47c8a05bc5404d76663b8ae39d9f3099cbd7cf18f648c01f06634fa709d74a83`.
The hashes independently computed after downloading both archives match their
checkpoint index. Structured job status is `complete`, and the resolved phase-B
configuration hash is
`1e3f60c573bac06eb42a1814e75f0707da8f30f42d5e0b2125f7d655bfcc7d13`.
Kaggle reported only 0.08 of 30 GPU hours used after all allocation and smoke
attempts. The resumability prerequisite was therefore satisfied.

The retained Colab staging utilities remain an alternative execution path, but
the authoritative E003 run used the private Kaggle datasets and kernels above.

## Training and Sinhala validation

The first full-training submission stopped before model loading because Kaggle
mounted the runtime dataset under a generated directory name. The second used
content-based discovery and completed all 500 steps on a Tesla T4. This failed
attempt consumed no optimizer step and is recorded separately.

Training took 2,965.58 seconds and canonical validation generation took 60.76
seconds. Peak allocated GPU memory was 1,803,472,384 bytes. Mean training loss
was 6.7171; five-step window loss fell from 22.33 at step 5 to 3.10 at step 500,
and all recorded gradient norms were finite. All 20 checkpoint archives from
step 25 through step 500 were downloaded and independently matched the hashes
in `checkpoint-index.json`. Final artifact hashes are:

- adapter: `7523d08297cf64eebc17cd16065d86b565e0d88f5b971f64a90eeab6f2f8d6bb`
- Sinhala predictions: `daaee600453ef85093a6a61bc7e13f7c82a22e06de37fd1ba91444e560a8907b`
- checkpoint index: `74da060b4d3be21b1820909705065e0f8e7dc2f781833ffd5369d3194aa41f91`

| Sinhala validation metric | E001 | E002 | E003 |
|---|---:|---:|---:|
| Strict WER | 114.26% | 98.24% | 95.87% |
| Strict CER | 83.79% | 42.42% | 39.48% |
| Canonical WER | 114.18% | 97.53% | 95.07% |
| Canonical CER | 83.86% | 41.43% | 38.23% |

Against E002, canonical WER fell 2.47 percentage points (paired 95% interval
-4.72 to -0.21) and canonical CER fell 3.20 points (-4.50 to -1.90). Word
substitutions/deletions/insertions changed by -17/-2/-5; character operations
changed by -82/-57/-20. At word level, 56 rows improved, 116 tied and 34
regressed. At character level, 109 improved, 49 tied and 48 regressed. E003
therefore passes its Sinhala-retention thresholds and produces a small,
statistically supported additional Sinhala gain, although 95.07% WER and 38.23%
CER remain far above the project's under-10% objective.

## English retention

E003 was evaluated with the same English prompt on the unchanged 2,620-row
LibriSpeech test-clean benchmark. Benchmark SHA-256
`eb1d6f299f5fefde5b66fab450ffbc3b5bf2518ec9e64d3829c050579c6f2906`
and prediction SHA-256
`4dd167a079c7422ff00a1bf50495b3f4840f595fedcc7256b8bd4880e4486ee8`
matched locally and remotely. Inference took 900.46 seconds on a Tesla T4.

| English metric | Untouched | E002 | E003 |
|---|---:|---:|---:|
| Strict WER | 98.83% | 99.29% | 22.10% |
| Strict CER | 100.38% | 100.12% | 14.82% |
| Canonical WER | 4.23% | 6.35% | 13.84% |
| Canonical CER | 1.92% | 2.89% | 5.88% |

The apparent strict-metric improvement is a formatting effect: the English
replay references are LibriSpeech's uppercase, unpunctuated transcripts, and
E003 often reproduces that style. Canonical normalization removes case and
punctuation, revealing substantially worse recognition content. Against
untouched Whisper-small, canonical WER increased 9.60 points (paired 95%
interval +9.15 to +10.04) and CER increased 3.96 points (+3.62 to +4.30).
Against E002, WER increased 7.48 points (+7.01 to +8.01) and CER increased 2.99
points (+2.63 to +3.36). Relative to untouched, 1,782 rows regressed versus 103
improved at word level; substitutions/deletions/insertions increased by
4,227/494/380. The English-retention failure is broad and conclusive.

## Conclusion and next controlled experiment

E003 answers its question negatively: a 10% replay mixture using raw
LibriSpeech references does not preserve Whisper's English capability. It
slightly improves Sinhala while teaching the decoder a different English output
style and degrading normalized word content. More raw-reference replay is not
the next justified move because it would strengthen the treatment that failed.

The next controlled experiment should keep E003's audio rows, ratio,
architecture, optimization, and Sinhala data fixed, but replace only the 1,111
English training targets with frozen untouched-Whisper teacher transcripts.
This behavior-replay target directly rehearses the English outputs being
preserved, including their casing and punctuation, while avoiding test-clean
leakage. Generate teacher transcripts only for the existing train-clean-100
replay audio, freeze their hashes before training, and call this E004. If E004
does not restore English retention without losing E003's Sinhala gain, the next
axis should be gradient isolation or separate language adapters rather than an
uncontrolled increase in replay volume.

## Decision rule

The thresholds are frozen before training. E003 must retain at least 75% of
E002's improvement over E001 on both canonical Sinhala metrics: WER at or below
101.69% and CER at or below 52.04%. Its English canonical WER may be at most
0.50 percentage points above untouched Whisper-small, and the upper end of the
paired 95% interval may be at most +1.00 point. Report exact paired confidence
intervals and error operations; do not select it from aggregate WER/CER alone.

The Sinhala branch passed, but the English branch failed by a wide margin.
E003 is rejected; its result must not replace the current candidate.
