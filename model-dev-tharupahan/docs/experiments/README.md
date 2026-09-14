# Experiment Registry

Every model experiment receives one permanent sequential ID. The same ID is
used for its notebook, configuration, run directory, generated predictions,
and result report.

| ID | Experiment | Status | Result report |
|---|---|---|---|
| E000 | Untouched Whisper-small on v4 validation | Complete | [E000](e000-whisper-small-zero-shot-v4.md) |
| E001 | Whisper-small wide LoRA rank 16, 100 steps, v4 pilot | Complete | [E001](e001-whisper-small-wide-lora-r16-100-step-v4.md) |
| E002 | Whisper-small wide LoRA rank 16, 500 steps, nested v4 data | Complete | [Sinhala improved; failed original English gate, passes current 10% ceiling](e002-whisper-small-wide-lora-r16-500-step-v4.md) |
| E003 | E002 recipe with 10% raw-reference English replay | Complete | [Sinhala improved; English retention failed](e003-whisper-small-wide-lora-r16-english-replay-v4.md) |
| E004 | E003 recipe with untouched-model teacher-target English replay | Complete | [English retention passed; Sinhala remained far above target](e004-whisper-small-wide-lora-r16-teacher-replay-v4.md) |
| E005 | E004 teacher replay scaled to 50k Sinhala rows and one effective epoch | Complete | [Sinhala improved materially; English retention passed](e005-whisper-small-wide-lora-r16-50k-teacher-replay-v4.md) |
| E006 | E005 recipe scaled to a nested 100-hour Sinhala tier | Complete | [Sinhala improved materially; English retention passed](e006-whisper-small-wide-lora-r16-100h-teacher-replay-v4.md) |
| E007 | E006 recipe scaled to the complete 220.877-hour v4 training split | Complete -- both gates pass; end of the nested data-scale curve | [Sinhala improved materially; English retention passed](e007-whisper-small-wide-lora-r16-full-v4-teacher-replay.md) |
| E008 | Automated LoRA rank/learning-rate search (Optuna) over the E001 recipe | Complete (search only, not a controlled experiment) | [Search suggests rank=32, lr~2.3e-4; not yet validated](e008-optuna-rank-lr-search-v4.md) |
| E009 | Sinhala tokenizer vocabulary extension (+250 tokens) on the E001 recipe | Stopped -- fix confirmed correct, technique still doesn't beat E010 at pilot scale | [Fix verified; corrected pilot still degenerate, does not beat E010](e009-tokenizer-extension-pilot-v4.md) |
| E010 | Controlled validation of E008's rank/LR search finding at 500 steps | Complete -- finding validated | [rank=32, lr~2.3e-4 wins on every metric](e010-rank-lr-validation-v4.md) |
| E011 | Zero-training bake-off: Meta Omnilingual ASR CTC 300M v2 vs. E007 | Complete for CTC 300M; loses as officially scored, but a real wrong-script anomaly masks the true result | [Official loss, but real evidence of a masked, likely-fixable script-conditioning issue](e011-omnilingual-ctc-bakeoff-v4.md) |
| E012 | Full-parameter fine-tune: LR x replay-ratio search (v5 data) | Complete -- lr=5e-5 wins decisively over lr=1e-6/5e-6; replay ratio (10/20/30%) barely differs on Sinhala-only signal, English retention not yet checked | [E012](e012-full-finetune-lr-replay-search-v5.md) |
| E013 | Score E007 against dataset v5's new test set (TTS+YouTube) and validation | Blocked -- 3 real bugs found and fixed, then Kaggle's weekly 30h GPU quota exhausted before a scored result | [E013](e013-e007-v5-scoring.md) |
| E014 | Bounded full-finetune LR/scheduler comparison with real English-retention evaluation | Complete -- both arms overfit the repeated pilot; both pass the current 10% English ceiling, enabling one checkpointed full-v5 epoch | [E014](e014-full-finetune-retention-validation-v5.md) |
| E015 | Full-parameter Whisper-small, one v5 training epoch selected on v6 validation | Preparing -- immutable inputs and exact-runtime smoke before phase A | [E015](e015-full-parameter-one-epoch-v5-v6.md) |

The fixed 2,620-row LibriSpeech test-clean benchmark is evaluated after every
adapter experiment without changing its references or normalization protocol.

## Naming rules

- Files use lowercase kebab-case, except Python modules which use snake_case.
- Experiment assets begin with `eNNN-` and use the same descriptive stem.
- Notebooks live only in `notebooks/`; prose copies of notebooks are not kept.
- Stable project/reference documents remain at the project root.
- Experiment reports live in `docs/experiments/`.
- Generated local results live in `reports/experiments/eNNN-.../`.
- Transfer bundles live in `reports/colab/` and include dataset version, role,
  row count, and `audio` when they embed audio.
- Superseded names are removed rather than retained as aliases.
