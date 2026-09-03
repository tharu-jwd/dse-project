# Transcript Review Provenance

Dataset v2 uses transcript corrections reviewed and approved by the project
owner using GPT-assisted suggestions. ChatGPT reviewed text only and did not
receive or evaluate the audio. Its returned TSV files were validated for exact
sample-ID coverage, row count, ordering, immutable original text, and non-empty
suggestions before application.

The GPT-assisted pass proposed changes to 294 of 2,000 gold candidates:

- 16 formatting, case, or punctuation-only changes
- 91 spacing-only changes
- 187 spelling or lexical changes
- 1,706 unchanged transcripts

The proposed text differs from the original references by 4.89% WER and 0.55%
CER when the two transcript versions are compared directly. This quantifies the
revision magnitude; it is not a model accuracy result.

Existing native audio-based decisions take precedence over text-only
suggestions; 11 such decisions were present when v2 was created. Every overlay
record retains `review_method`, the original transcript,
corrected transcript, suggestion confidence/class where applicable, timestamp,
and notes. Raw source files remain unchanged.

The finalized v2 fingerprint is
`1001c6cb3450de825a6764b53486b0a0add1beca375156d703771a908f1e7954`.
It contains 182,665 train, 1,000 validation, and 999 test rows. One reviewed
candidate was excluded as uncertain, in addition to the two automatic duration
exclusions. Relative to v1, 289 final validation/test canonical transcripts
changed. Train/validation/test have zero speaker overlap.

This review improves spelling and formatting consistency but cannot detect
audio/transcript mismatch, omitted speech, or GPT standardization that differs
from what was spoken. Those remain limitations of v2 and should be considered
when interpreting validation and test metrics.
