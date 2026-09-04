# Training Configuration Layout

- `experiments/`: active or completed numbered model experiments
- `diagnostics/`: tiny smoke tests and historical controlled diagnostics

Numbered experiment configurations use the same `eNNN-` identifier as their
notebook, report and generated run directory. Diagnostic filenames retain their
historical identities so old results remain reproducible.
