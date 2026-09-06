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

## Downloading data directly on the job beats uploading it

The v4 manifest's train/validation splits are drawn entirely from OpenSLR-52
(14GB of the 17GB `data/raw/`), and OpenSLR-52 is a public dataset with its
own downloader already in this project
(`scripts/data/download_openslr52.py`, pulling from
`https://openslr.trmal.net/resources/52`). Measured upload speed from this
project's home connection to Camber Stash: ~2MB/s. Measured download speed
from the OpenSLR mirror *to a Camber job itself*: ~21MB/s (confirmed via a
live `curl` inside a job) -- roughly 10x faster, because it's a datacenter-
to-datacenter transfer rather than a home-upload-bandwidth-limited one.
Uploading the corpus once still has a place (any future job can then skip
the download entirely, since a `--path` mount already contains whatever the
last job in that stash location wrote there -- see `--path` note above), but
for a first run, having the job download its own input data beat uploading
it by close to an order of magnitude.

## A completed job's output silently accumulates in the same `--path`

Confirmed twice now, both times causing the *next* job against that same
`--path` to sit `PENDING` for 20-40+ minutes (versus a normal 1-3 minutes)
with no error, no explanation, and no way to see why:

1. An aborted local-upload attempt left ~48,000 partial files under
   `data/raw/openslr52/...` in a stash path before a later job mounted it.
2. A *successful, completed* job's own output (`runs/<name>/checkpoint-*/`,
   hundreds of MB of adapter/optimizer state per checkpoint) synced back
   into the same stash path automatically -- and the next job that mounted
   that path (unrelated to the first job, just sharing `--path`) sat stuck
   the same way.

Neither case produces any diagnostic -- the job just never leaves
`PENDING`. The fix both times: `camber stash rm -rf` the accumulated
directory, then submit a fresh job (there is no cancel command for the
stuck one -- it likely never resolves, but a new submission against the
now-clean path starts normally within 1-3 minutes). This does **not**
retroactively fix an already-submitted stuck job.

Practical rule: after any job that produces real output completes, clean
its output out of the shared `--path` stash location (download what's
needed first) before submitting the next job against that same path --
don't wait to notice the next job stalling.

## GPU-hour budget: running total

The 5 included hours are not exposed by any CLI command (see above) -- this
project tracks them manually here, computed from `camber job get --output
json`'s `created_at`/`finished_at` on every `--gpu` job (non-GPU jobs are
free and excluded).

| Job | Purpose | Duration |
|---|---|---|
| `25175` | GPU/PyTorch smoke test | 4.7 min |
| `25184`-`25194` | E009 pilot, 6 failed attempts (see below) | 49.7 min |
| `25195` | E009 pilot, succeeded | 9.9 min |
| `25196` | E008 search, stuck `PENDING` (accumulated-output bug above), abandoned | unknown -- never left `PENDING`, likely free (billing appears tied to `RUNNING`, not confirmed) |
| `25199` | E008 search, resubmitted against cleaned path | in progress |
| **Total, excluding `25196`'s unknown cost (2026-09-06)** | | **~1.07 hours + `25199`** |
| **Remaining of the 5-hour budget** | | **~3.93 hours, minus `25199`** |

Update this table (recompute from `camber job get <id> --output json` for
every `--gpu` job since the last entry) whenever a new Camber job runs, not
just when a batch of work finishes -- it is easy to lose track across a long
session otherwise, which is exactly what happened before this table existed.

## Verified end-to-end (2026-09-06)

The E009 tokenizer-extension pilot was the first real (not smoke-test) job,
and took **7 attempts** before succeeding -- each failure was a genuine,
previously-unknown environment gap, not a repeat of the same mistake:

1. `25184`: `pyproject.toml` requires Python >=3.11; this image's plain
   `pip` pointed at a 3.10.12 interpreter. Fixed with `--ignore-requires-python`.
2. `25185`: with that flag, downloaded the full corpus (~19 min) then hit
   `ModuleNotFoundError: No module named 'numpy'` -- bare `pip install` and
   `python3` resolved to *different* interpreters on this image (a system
   3.10.12 vs. a Spack-built 3.11.7). Fixed with `python3 -m pip install`.
3. `25189`: with the interpreters aligned, pip resolved the latest numpy
   (2.5.2), which requires Python >=3.12 -- not available on this image
   either. Fixed with an explicit `numpy<2.2` pin on the install command
   (not in `pyproject.toml` -- this is a Camber-image-specific constraint,
   not a real project dependency change).
4. `25191`: downloaded the corpus a *second* time (the script has no
   already-extracted check, see below) then failed on
   `configs/training/experiments/e009-tokenizer-extension-pilot-v4.json` not
   existing on stash -- it was created locally after the initial bulk code
   upload and never re-synced.
5. `25192`: (download step dropped once the corpus was confirmed already
   synced back to stash from job 4's run, despite that job's overall
   failure -- see the `--path` note above) failed on the same missing-file
   class of bug again, for `TrainConfig`'s `extended_tokenizer_path` field
   and the tokenizer artifact itself.
6. `25194`: after a full re-sync of `src/`, `scripts/`, `configs/training/`,
   failed on `git rev-parse HEAD` -- `train.py`'s `git_commit()` had only
   ever run inside a real git checkout before; a stash-populated workdir has
   none. Fixed by making it degrade to `"unknown"` instead of crashing (see
   `docs/audits/e009-tokenizer-extension.md`).
7. `25195`: succeeded -- ran to completion. (Its *training result* was still
   bad, for an unrelated, model-architecture reason -- frozen embeddings
   under LoRA -- documented in the E009 audit, not a Camber issue.)

Practical lesson for the next Camber job: sync every locally-changed file to
stash right before submitting (an easy thing to forget mid-session -- 3 of
these 7 failures were exactly this), and expect the underlying image's
Python/pip/numpy versions to need explicit handling rather than assuming
parity with any other environment this project has used.
