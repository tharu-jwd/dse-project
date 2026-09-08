# E013 — Score E007 against dataset v5's new test set

## Status

**Blocked on Kaggle's weekly GPU quota (30h), exhausted 2026-09-08.** Three
real bugs were found and fixed in sequence before hitting this hard external
limit; none of the three attempts produced a scored result. Resume once the
quota resets.

## Question

Items 1/2 from the plan: how does E007 (current best model, LoRA rank 16,
trained on v4's full 220.877h) perform on dataset v5's new test set (Path
Nirvana TTS + SPEAK-ASR YouTube, 22.72h, zero OpenSLR -- data E007 has never
seen in training or evaluation), and on a 200-row slice of v5's validation
set (52 held-out speakers)? Uses beam5 (the project's own recommended
default per plan.md item 1) via `predict.py`'s new `--adapter`/`--num-beams`
support.

## Attempts

1. **Version 1 (fast, cheap failure, ~1 min):**
   `RuntimeError: expected one predict.py, found [...]` --
   `one_file("predict.py")` matched two files: the actual script and
   `sinhala_asr/evaluation/predict.py`, a same-named module inside the
   bundled package. The same latent ambiguity existed for
   `one_file("adapter_model.safetensors")` -- the E007 phase-b kernel
   source carries several intermediate `checkpoint-N/adapter_model
   .safetensors` files alongside the real `final-adapter/` one. Fixed both
   by anchoring on fixed, known relative paths instead of a name search --
   `predict.py` as `sinhala_asr`'s sibling, `final-adapter/` via the run's
   own small, unique result marker (`e007-phase-b-kaggle-result.json`),
   matching the convention `kaggle/e007-decoding-comparison/e007-decoding-
   comparison.py` already established for this exact adapter lookup.
2. **Version 2 (fast, cheap failure, ~1 min):**
   `FileNotFoundError: data/raw/sinhala-tts-pathnirvana/data.parquet` --
   `predict.py`'s manifest rows store `source_path` as relative paths
   matching each dataset's own local layout; without `cwd` set to that
   dataset's root, pyarrow can't resolve them. Same class of bug E012's
   search kernel had already fixed once, missed here on the first pass.
   Fixed by setting `cwd=str(manifest.parent)` per run (test and
   validation manifests live in two different Kaggle datasets, so this
   can't be a single fixed directory).
3. **Version 3 (expensive failure, 5.29h):** ran successfully through all
   6,386 Path Nirvana TTS rows (13.61h of audio, `cwd` fix confirmed
   correct) before crashing on the very first YouTube row:
   `FileNotFoundError: /Users/tharupahan/Code/dse-project/model-dev-
   tharupahan/data/raw/youtube/data/train-00000-of-00002.parquet` -- an
   absolute local Mac path. `youtube-upstream`'s manifest was built in an
   earlier session, before this session's relative-path Kaggle-transport
   convention existed, and stored absolute `source_path` values. Fixed
   generically in `scripts/data/build_external_eval_manifest.py`: strip
   this repo's own root prefix from any `source_path` that starts with
   it, so any future source with the same problem gets fixed the same
   way, not just youtube specifically. Re-uploaded the corrected manifest
   as a new version of `sinhala-asr-e013-test-scoring-inputs` (~2.2GB,
   only the small manifest file actually changed).

Real, useful signal even without a scored result: **beam5 batch=8
inference of the 13.61h Path Nirvana TTS portion took 5.29h wall-clock on
a T4** -- roughly 2.6x faster than real-time, not the 1-2h this project
first guessed. Scoring the full 22.72h test set at this rate would need
roughly 9h of GPU time in one run; the 200-row validation slice is small
and cheap by comparison.

4. **Push blocked:** `Kernel push error: Maximum weekly GPU quota of 30.00
   hours reached.` Cannot resubmit until Kaggle's weekly quota resets --
   no code fix available for this one. E012's investigation (six
   attempts, ~1h40min crashed run + a full ~4-5h successful 9-trial run)
   plus these three E013 attempts (two fast, one 5.29h) account for most
   of the week's usage.

## Next step

1. Wait for Kaggle's weekly GPU quota to reset, or use a different GPU
   source if the owner wants to proceed sooner.
2. Resubmit `kaggle/e013-e007-v5-scoring/` as-is once quota is available --
   all three known bugs are fixed; the job should run clean based on
   version 3's clean TTS-portion result.
3. Given beam5's real ~9h cost for the full test set, consider whether a
   smaller representative test subset (or greedy decoding, at the cost of
   the beam5 WER improvement) is worth trading for a faster first look,
   before committing a full 9h GPU session to it.
