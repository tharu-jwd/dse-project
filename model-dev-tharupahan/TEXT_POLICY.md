# Sinhala Transcript Policy v1

The implementation version is `si-conservative-v1`. Three representations are
kept: immutable `text_original`, deterministic training `text_canonical`, and
scoring-only `text_metric`. A native correction is stored separately as
`text_reviewed`; it never overwrites the upstream transcript.

## Canonical training text

- Normalize Unicode to NFC.
- Collapse all whitespace runs to one ASCII space and trim the ends.
- Remove spacing immediately before `, . ; : ! ? %`.
- Preserve punctuation, Sinhala/Latin case as written, digits, English words,
  and Sinhala–English code-switching.
- Preserve ZWJ only in recognized Sinhala conjunct contexts after virama and
  before `ර`, `ය`, or `ෂ`; flag and remove other format/control characters.
- Do not merge or split compounds, rewrite particles, expand abbreviations,
  convert numbers/dates, transliterate, or choose among spelling,
  colloquial/formal, or dialect variants automatically.

## Canonical metric text

Start from canonical text, lowercase cased scripts, replace Unicode punctuation
with spaces, and collapse whitespace. Do not remove digits, transliterate
English, or change word segmentation. This view helps distinguish formatting
errors but never replaces strict scoring.

## Human decisions

Use `correct` only when the transcript matches the audible speech under this
policy. Use `edited` and enter exactly what was spoken when wording is wrong.
Use `bad_audio`, `mismatch`, `duplicate`, or `uncertain` instead of guessing.
English words spoken inside Sinhala audio remain in the reference. Standalone
English rows remain labeled `latin_only`; mixed Sinhala/Latin rows are labeled
`code_switched`.

Changing any rule requires a new normalization version, updated tests, a new
dataset fingerprint, and rescoring every compared prediction set.
