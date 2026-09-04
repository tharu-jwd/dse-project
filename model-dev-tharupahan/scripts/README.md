# Script Layout

- `data/`: download, index, audit, version and finalize datasets
- `review/`: build and operate human-review and audio-analysis workflows
- `evaluation/`: run ASR inference and calculate model/refiner metrics
- `training/`: train models and build/run Colab training bundles
- `refinement/`: historical transcript-refinement experiments

Scripts are grouped by responsibility. Reusable implementation belongs in
`src/sinhala_asr/`; scripts should remain thin command-line entry points.
