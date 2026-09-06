# Camber Cloud: setup and verified operating notes

## Why Camber, and what it's for

Camber Cloud is a third compute option alongside local MPS and Kaggle,
available through a GitHub Student Pack grant (5 free GPU hours, this
project's first use of the platform). It is not a replacement for
Kaggle -- Kaggle remains the platform for full controlled experiments
(E003-E007) because its free weekly GPU-hour allowance is much larger. Camber
is for GPU-bound work that doesn't fit either of the other two lanes: bigger
than what local MPS can do in reasonable time, but not worth spending scarce
Kaggle quota on before it's de-risked (for example, a bounded pilot for the
tokenizer vocabulary extension, item 3 in
[the priority order](../project/plan.md)). Given the 5-hour budget, treat it
as a place to run a small number of short, well-defined jobs, not a
replacement execution platform.

## Install and auth

```
curl -fsSL https://cli.cambercloud.com/install.sh | bash   # reviewed before running; see below
source ~/.zshrc   # or open a new shell -- the installer appends to PATH but doesn't export it into the current one
camber login       # interactive browser OAuth; cannot be completed headlessly
camber me          # verify: prints username, email, stash root
```

The installer downloads a versioned binary tarball from
`https://cli.cambercloud.com/releases/<version>/...` and installs to
`~/.camber/bin`, updating `.zshrc`'s `PATH`. It was read in full before being
run in this project (167 lines, plain OS/arch detection + curl download +
`chmod +x` + atomic `mv` into place -- no surprises). `camber login` opens a
browser; it must be run in the user's own terminal, not from an agent.

The most common failure after a fresh install is `command not found: camber`
in the *same* shell the installer ran in -- the `PATH` export only takes
effect in new shells. `source ~/.zshrc` (or `export PATH="$HOME/.camber/bin:$PATH"`
directly) fixes it without needing a new terminal window.

## CLI surface (verified via `--help`, not assumed from docs)

Top-level commands: `agent, app, chat, engine, job, login, me, stash, tag,
team, update, version`. The two that matter for running training jobs:

- `camber job create --cmd "<shell command>" --engine base --path stash://<user>/<dir>/ --size <size> [--gpu] [--num-nodes N]`
  -- submits a job. `--path` is required even for a job that needs no input
  data; it becomes the job's working-directory mount point (verified via
  `camber job get <id> --output json`: the stash path's last segment becomes
  `mount_dir` inside the container).
- `camber job get <id> [--output json]` -- status (`PENDING` / `RUNNING` /
  `COMPLETED` / `FAILED`); the JSON form also reports the underlying
  container image and tag actually used.
- `camber job logs <id>` -- stdout/stderr of a finished (or running) job.
- `camber stash ls / mkdir / cp / rm / test` -- file management; `cp -r` for
  directories, `--exclude` / `--use-gitignore` when uploading a local project
  folder.

There is no separate "get my remaining GPU-hour balance" command in this CLI
version -- track the 5-hour budget manually (or via Camber's own web
dashboard), not via `camber job list`/`get`, which report per-job status
only, not cost.

### `--gpu` restricts node size

`--size` accepts `xxsmall, xsmall, small, medium, large` for CPU jobs, but a
`--gpu` job **only accepts `xsmall` or `medium`** -- passing `xxsmall` (the
CLI's own displayed default) with `--gpu` fails outright:

```
Error: invalid node size for GPU: xxsmall. Must be one of: [xsmall medium]
```

`xsmall --gpu` provisions a single NVIDIA L4 (23034MiB), confirmed via a live
`nvidia-smi` run inside a submitted job -- this matches the $1.50/hr tier from
Camber's published pricing.

## The `base` engine does not ship PyTorch by default

`camber engine list` describes the `base` engine as "include sklearn,
pytorch, pandas, matplotlib, pytorch, MPI and more". This does not hold in
practice: a job run as `python3 -c "import torch"` on `base` fails with
`ModuleNotFoundError: No module named 'torch'`, confirmed directly (not
inferred from the description). A follow-up probe job on the same engine
found:

- `python3` resolves to a Spack-built interpreter
  (`/opt/spack/.../python-3.11.7-.../bin/python3`), not a conda environment
  -- `conda: command not found`.
- Of `torch`/`transformers`/`numpy`/anything `nvidia`-named, only `numpy`
  (2.1.0) is preinstalled.
- The GPU and CPU variants of the `base` engine use different pinned
  container images (`hpc-base:cuda70-v2024.8.20` for `--gpu` jobs vs
  `hpc-base:mpi5.0-cpu-v2025.6.19` without), so don't assume parity between
  what a CPU probe job and a GPU job each have installed.

Practical consequence: any real job needs its own `pip install torch
transformers peft ...` (or an equivalent pinned environment step) as part of
`--cmd`, exactly like preparing a fresh Kaggle kernel -- there is no shared
prebuilt ML image to rely on. Budget the install time into the job's expected
GPU-hour cost, since `--gpu` jobs are billed from provisioning, not from when
training actually starts.

## Verified end-to-end (2026-09-06)

Two smoke-test jobs were run before committing any real work to the
platform, following this project's standing rule of verifying infrastructure
before trusting it with anything that matters:

- Job `25175` (`--gpu --size xsmall`): confirmed a real, healthy L4 is
  reachable and `nvidia-smi` reports correctly; confirmed the missing-torch
  finding above. Runtime (submit to finished): ~4m40s, well under the 5-hour
  budget.
- Job `25176` (`--size xxsmall`, no `--gpu`, free of the GPU-hour budget):
  confirmed the Python/conda/package findings above.

No training job has been run on Camber yet -- this document exists to make
the next one (whenever there's a concrete GPU-bound task ready, most likely
the tokenizer-extension pilot) start from verified facts instead of the
platform's own documentation, some of which (the WebFetch-inaccessible docs
site aside) turned out not to match the actual running environment.
