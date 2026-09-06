# E008 eval-time repetition bug (found and fixed before the search continued)

Evidence grade: **Verified artifact** — reproduced directly by loading the
affected checkpoint and comparing generation output with and without the fix,
not inferred from metrics alone.

## What was observed

The E008 rank/LR Optuna search (`scripts/training/optuna_search.py`, config
`configs/training/experiments/e008-rank-lr-search-base-v4.json`, 100 steps per
trial) ran its first trial (`lora_rank=32, learning_rate=3.42e-5`) to
completion:

```
train_loss: 7.865 (started at 11.85 step 5, decreasing monotonically to 6.99 step 100 - normal)
eval_loss:  1.741 (1.825 at step 50 -> 1.741 at step 100 - normal, decreasing)
eval_wer:   1.002
eval_cer:   3.463  (346%)
```

`eval_loss` and the training loss curve behaved exactly as expected for
early-stage LoRA fine-tuning: monotonic decrease, sane gradient norms
(6-18), no divergence. But `eval_cer` of 346% is not something normal
transcription errors produce — a wrong-but-plausible-length hypothesis tops
out CER somewhere under 200% even when badly wrong. A CER this high is the
signature of the hypothesis being far *longer* than the reference, i.e. a
decoding failure, not a learning failure.

## Root cause (confirmed, not assumed)

`scripts/training/train.py`'s eval loop (`predict_with_generate=True`) uses
greedy decoding with `generation_max_length=225` and no repetition penalty of
any kind. Loaded the trial's own `checkpoint-100` adapter directly (bypassing
the trainer) and called `model.generate()` on three validation clips:

```
REF: ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු
HYP (225 tokens, hit the cap): මුත්න්න්න්න්න්න්න්න්න්න්...  (single syllable repeated ~100x)

REF: හසන්ත සහෝගේ උපසිරැසි
HYP (225 tokens, hit the cap): මුරුමුමුමුමුමුමුමුමුමුමු...  (same pattern)
```

Setting only `model.generation_config.no_repeat_ngram_size = 3` (no other
change, same checkpoint, same three clips) eliminated the loop entirely —
outputs terminated naturally at 24-42 tokens instead of running to the cap:

```
REF: ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු
HYP (42 tokens): මුත්න් නිතයි කිස්වානයක්
```

Still wrong (100 steps is not enough training to produce a correct
transcription — that part is expected), but no longer a degenerate loop, and
no longer capable of pushing CER past 100%.

This is the same failure mode already known and handled in
`scripts/review/try_model_app.py`'s `transcribe()` (which passes
`no_repeat_ngram_size=3` at call time) — it just hadn't been ported back into
`train.py`'s own eval path, because every prior experiment (E000-E007) trained
far past the point where an undertrained checkpoint babbles, so their
*final*-checkpoint eval numbers never hit this regime. It only surfaced now
because E008 deliberately evaluates checkpoints at a tiny, fixed step budget
by design.

## Why this mattered for the search specifically

`optuna_search.py` minimizes `eval_wer`. If every trial in the 8-trial search
saturates near `eval_wer ~= 1.0` because every undertrained checkpoint
degenerates into a repetition loop under greedy decoding, the search has no
signal to discriminate between rank/LR combinations — Optuna would pick
whichever trial happened to have a marginally lower value inside that noise
floor, not the genuinely best hyperparameters. Continuing the original
8-trial run (already budgeted at ~75 min/trial, ~10 hours total) would have
spent that time producing a result that looked complete (a study.db with a
"best trial") but wasn't a valid signal.

## Fix applied

Added `model.generation_config.no_repeat_ngram_size = 3` to
`scripts/training/train.py`, next to the existing `language`/`task`/
`forced_decoder_ids` generation_config assignments (same mechanism already
proven to take effect at eval time in this file — that's how `language="si"`
correctly drives Sinhala-only decoding for every past experiment). Verified
the assignment alone (no generate()-call kwargs) suppresses the loop, i.e. it
really does flow through `Seq2SeqTrainer`'s `predict_with_generate` path the
same way the other generation_config fields already do.

Did **not** touch `scripts/training/run_e002_colab.py` (the shared Kaggle/
Colab runner used by E005/E006/E007) even though it has the same gap,
because:
- E005 and E006 both completed with sane final WER/CER (well under 100%),
  which is only possible if their final checkpoints were past the babbling
  regime — so this bug does not put either experiment's *reported* (final)
  result in question.
- E007 is a currently-running Kaggle job; editing that script would require
  restarting a multi-hour in-flight run for a change that only affects
  interim eval-step numbers, not the final one that matters.
- Apply the same one-line fix to `run_e002_colab.py` before the *next* new
  Kaggle kernel is written (tokenizer-extension pilot or any future
  short-step Kaggle run would be exposed to the same issue).

## Disposition

- E008's stale `runs/e008-rank-lr-search/` (trial-000, trial-001, study.db)
  deleted; trial 0's `eval_wer=1.002` result is invalid and was discarded,
  not counted.
- Search relaunched from a clean study with the fixed `train.py`.
