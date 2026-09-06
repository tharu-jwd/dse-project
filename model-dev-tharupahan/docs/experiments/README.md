# Experiment Registry

Every model experiment receives one permanent sequential ID. The same ID is
used for its notebook, configuration, run directory, generated predictions,
and result report.

| ID | Experiment | Status | Result report |
|---|---|---|---|
| E000 | Untouched Whisper-small on v4 validation | Complete | [E000](e000-whisper-small-zero-shot-v4.md) |
| E001 | Whisper-small wide LoRA rank 16, 100 steps, v4 pilot | Complete | [E001](e001-whisper-small-wide-lora-r16-100-step-v4.md) |
| E002 | Whisper-small wide LoRA rank 16, 500 steps, nested v4 data | Complete | [Sinhala improved; English-retention gate failed](e002-whisper-small-wide-lora-r16-500-step-v4.md) |
| E003 | E002 recipe with 10% raw-reference English replay | Complete | [Sinhala improved; English retention failed](e003-whisper-small-wide-lora-r16-english-replay-v4.md) |
| E004 | E003 recipe with untouched-model teacher-target English replay | Complete | [English retention passed; Sinhala remained far above target](e004-whisper-small-wide-lora-r16-teacher-replay-v4.md) |
| E005 | E004 teacher replay scaled to 50k Sinhala rows and one effective epoch | Complete | [Sinhala improved materially; English retention passed](e005-whisper-small-wide-lora-r16-50k-teacher-replay-v4.md) |
| E006 | E005 recipe scaled to a nested 100-hour Sinhala tier | Complete | [Sinhala improved materially; English retention passed](e006-whisper-small-wide-lora-r16-100h-teacher-replay-v4.md) |

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
