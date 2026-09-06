# E011 — Zero-training Omnilingual ASR CTC bake-off vs. E007

## Status

**Complete for CTC 300M v2; CTC 1B v2 and the LLM-ASR variants not run**
(Camber GPU-hour budget ran low after six real environment-bug attempts
to get the library running at all -- see
[camber-cli.md](../training/camber-cli.md) for the full trace). As
officially scored with this project's own metrics, CTC 300M v2 loses
decisively to E007. But a real, quantified anomaly in the result --
most predictions come back in the wrong Brahmic script, not the
requested Sinhala one -- means this number should not be read as "the
model can't do Sinhala." See "The real finding" below before drawing a
conclusion from the headline number alone.

## Question

Item 3 in [the plan's ranked execution order](../project/plan.md#3-run-a-zero-training-model-family-bake-off):
does Meta's Omnilingual ASR (CTC, wav2vec2-based, avoids Whisper's
Sinhala byte-token bottleneck) outperform E007's Whisper-small LoRA on
identical audio, references, and normalization, with zero training cost?

## Frozen design

- Candidate: `omniASR_CTC_300M_v2` via the library's own
  `ASRInferencePipeline`, `lang=["sin_Sinh"]`, `dtype=torch.float32`
  (not the library's bf16 default -- matches this project's established
  practice on Turing-class GPUs, though this run used an L4/cu128 node).
- Control: E007's already hash-verified Sinhala validation predictions
  (`f9645dc24424549b262a50382a4db6d1e3a97a19281f0ebd57cffa88bf79708b`),
  scored with the same metrics module -- not re-run, since E007 was
  already scored against this exact 206-row validation set.
- Identical audio/references: loaded directly from
  `data/versions/v4/manifest.parquet`'s own `validation` split via this
  project's own `load_training_rows`/`ManifestAudioDataset` (not a
  separately exported bundle), so "identical audio, references,
  normalization" holds by construction. Confirmed directly: 0 reference-
  text mismatches between the two prediction sets over all 206
  `sample_id`s.
- Platform: Camber Cloud (`scripts/evaluation/run_omnilingual_bakeoff.py`),
  not Kaggle -- two real Kaggle environment failures (numpy ABI mismatch,
  then a torch/torchaudio CUDA mismatch) led to moving this one library
  to Camber's from-scratch environment; see kaggle-cli.md and
  camber-cli.md for both full traces.

## Getting it running: six real environment failures

Documented in full in [camber-cli.md](../training/camber-cli.md); summary:

1. Missing `soundfile` (this script's own runtime need, not installed).
2. `sinhala-asr`'s own `pyarrow<20` pin directly conflicts with
   `omnilingual-asr`'s declared `pyarrow>=20.0.0` -- unresolvable
   together; fixed by never pip-installing this project's own package at
   all (it only needs to be importable via `PYTHONPATH=src`).
3. `torchaudio`'s compiled extension needs `libcudart.so.13`; a
   diagnostic-only job confirmed the environment's `torch` resolved to a
   CUDA 12.8 build with only `libcudart.so.12` present anywhere -- a real
   CUDA-tag mismatch between independently-resolved `torch` and
   `torchaudio`, not a `LD_LIBRARY_PATH` visibility problem (confirmed by
   testing that fix first; it made no difference). Fixed by pinning both
   `torch==2.8.0` and `torchaudio==2.8.0` from the same explicit `cu128`
   index before installing `omnilingual-asr`.
4. A remaining shape bug: this script's own defensive `(1, N)`
   channels-first reshape of the audio array was itself wrong (the
   library's channel/time transpose logic only runs when resampling is
   needed, which it never is here since this project's audio is already
   16 kHz) -- fixed by passing the plain 1-D array the library's
   dict-input path actually expects.

Camber GPU-hour cost for the whole investigation: **~4.35 of the 5.00-hour
free budget**, across 8 jobs (one diagnostic-only). See camber-cli.md's
budget table for the per-job breakdown.

## Official result (as scored)

Hash-verified: `predictions-omniASR_CTC_300M_v2.parquet`
sha256 `151c0cd2f312d556d1b16aaff93263bf2fa64d8a1c829161c3388257b3a90819`.
Independently re-scored locally with this project's own
`evaluate_rows`/`aggregate` (`metric_normalize`, same methodology as every
other experiment):

| Metric | E007 (control) | CTC 300M v2 |
|---|---:|---:|
| Canonical WER | 81.71% | **98.97%** |
| Canonical CER | 26.15% | **94.21%** |

Paired bootstrap 95% intervals (CTC 300M minus E007, canonical, 2,000
iterations, same 206 rows matched by `sample_id`):

- WER delta: +12.4 to +22.0 percentage points -- excludes zero, real and
  decisive.
- CER delta: +61.1 to +75.0 percentage points -- excludes zero, real and
  decisive.

**As officially invoked, CTC 300M v2 loses decisively to E007.** Do not
stop reading here, though -- the reason is not what it looks like.

## The real finding: systematic wrong-script output, not a quality failure

A qualitative spot-check of raw predictions (never skip this step --
see [the project's evaluation protocol](../evaluation/evaluation.md))
showed something the aggregate number alone completely hides: most
predictions come back in **Bengali script**, not Sinhala, despite
`lang=["sin_Sinh"]` being passed on every single item. A few rows mix in
Kannada, Telugu, or Hangul characters. Quantified directly, over all 206
rows (Unicode block classification per character):

| Output script | Rows | Share |
|---|---:|---:|
| Pure Sinhala | 23 | 11.2% |
| Pure Bengali | 145 | 70.4% |
| Mixed (2+ scripts) | 30 | 14.6% |
| Other (pure Latin, Telugu, etc.) | 8 | 3.9% |

Sinhala and Bengali are both Brahmic-derived scripts with related
Indo-Aryan phonology -- a plausible, though unconfirmed, explanation is
that `lang` conditioning is weak or partially ignored in this simple
pipeline wrapper, and the model falls back to whichever related,
higher-resource language it is more confident in. No existing GitHub
issue on the upstream repo was found describing this specific behavior
(checked, not assumed).

Critically, the 23 rows that *did* stay in pure Sinhala script are not
low-effort or lucky exact matches -- scored on their own (a small,
non-random subset, not a valid standalone quality estimate, but
informative):

| Metric | This 23-row subset | E007 (same 23 rows' WER/CER, for scale) |
|---|---:|---:|
| Canonical WER | **67.16%** | worse than E007's own 81.71% aggregate |
| Canonical CER | **14.68%** | better than E007's own 26.15% aggregate |

Representative examples (hash-verified against the scored parquet above):

```
REF: වනජීවී නිලධාරීන් ස්ථානගතව
HYP: වනජීවි නිල්ධාරින ස්ථාන ගතව

REF: අමරා ඉරංගනී බුදු හාමුදුරුවන්ටත් ආරක්ෂක භටයෝ හිටියද?
HYP: අමරා ඉරංගනी බුධු ආමදරුවන්ටත් ආරක්ෂක බටೆෝ හිටියද

REF: මගේ ජීවිතය අද බොහොම යහපත් විදිහට ගත කෙරුණා.
HYP: මගේ ජීවිතේ අද භොහෝම හපත්විදිහට ගතකෙරුණා
```

These are close, substitution-heavy errors on real Sinhala words -- a
qualitatively different regime from the ~99%/94% aggregate, and a
different regime from E011's own wrong-script majority. This is real,
promising evidence that this model's underlying acoustic/phonetic
modeling of Sinhala may be substantially better than the official number
suggests, gated behind an apparent inference-time
language-conditioning issue rather than a fundamental capability gap.

## Conclusion and next step

**Do not advance to item 4 (a bounded Omnilingual CTC 300M adaptation
pilot) on the strength of this result alone**, in either direction: not
as a rejection (the official number is real but likely does not reflect
the model's true ceiling), and not as an endorsement (11.2% in-script is
not yet evidence of a fix, only evidence a fix might exist). Recommended
next step, deliberately kept cheap and zero/near-zero-cost since the
Camber budget is nearly exhausted (~0.6 hours of the original 5.00-hour
grant remain, see camber-cli.md): investigate the pipeline's actual
language-conditioning mechanism (does `ASRInferencePipeline` accept a
stronger per-item language constraint than the simple `lang` list already
tried? does batching multiple languages together dilute conditioning
versus a single-language batch? does the 1B variant show the same
script-drift rate?) before spending any further GPU budget -- this is a
software/API question first, not a training question, and does not need
another paid GPU job to make progress on.

CTC 1B v2 and the LLM-ASR variants from the original bake-off scope
were not run; revisit only after the script-drift question above is
understood, since the same issue would likely recur and waste budget on
the larger, slower model the same way.
