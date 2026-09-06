# Kaggle CLI Operations and Storage Safety

## Why Kaggle, and since when

Colab was the original platform (E000-E002). Starting with E003, Colab's free
T4 allocation returned four consecutive `503 Service Unavailable` errors
across three work cycles with no session ever created -- an infrastructure
failure, not a training failure, but one that left no usable path forward.
Kaggle (a separate provider, its own weekly free GPU-hour allowance) has been
the sole execution platform since E003, through E004-E007. The historical
Colab policy remains in [colab-cli.md](colab-cli.md) for reference; this
document describes actual current operations.

## Job structure

A Kaggle job is a private kernel: one Python script plus a `kernel-metadata.json`
declaring `dataset_sources` (private datasets mounted read-only under
`/kaggle/input/<owner>/<slug>/...`) and `kernel_sources` (another kernel's own
output, mounted the same way, used to chain a training kernel's result into a
downstream evaluation kernel without re-uploading anything). Prepare inputs
locally, hash-verify them, upload as a private dataset, push the kernel, then
poll for completion and download+verify the result -- the same
prepare/verify/upload/verify/run/verify discipline as Colab, adapted to
Kaggle's primitives.

## The `one_file()` mount-collision hazard

Training scripts commonly locate a mounted asset with a helper like:

```python
def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]
```

This breaks if the same filename exists in more than one attached dataset --
which happens easily, because it is normal for a runtime dataset and an inputs
dataset to each independently stage a copy of the same shared runner script.
E005's first kernel version crashed exactly this way (`run_e002_colab.py`
present in both an inputs and a runtime dataset) before any GPU time was
spent. The fix is not to remove a dataset attachment to eliminate the
duplicate -- that can silently remove a file the script needs for something
else entirely, which is exactly what happened on the first fix attempt for
that same bug, breaking a different `one_file()` call instead of solving the
original one. The correct fix is to anchor `one_file()` lookups on filenames
that are unique to a single attached dataset, verified locally against the
real dataset mirrors before pushing:

```bash
find <local-mirror-of-each-attached-dataset> -iname "<anchor-filename>"
```

Every anchor filename a kernel relies on should return exactly one hit across
the union of everything the kernel actually attaches, checked before every
push, not assumed from a previous kernel's success.

## Polling discipline

`kaggle kernels status <slug>` hits a rate limit that, once triggered, can
stay in effect for hours across an entire account -- not just the offending
process. `kaggle kernels output <slug> -p <dir>` is a more reliable
completion signal in practice: it returns cleanly with zero downloaded files
while a kernel is still running, and only produces files once the kernel
reaches a terminal state (a crash produces a log within seconds; a genuine
success produces the full output set). Prefer polling with `kernels output`
over `kernels status`.

Poll on an interval of minutes, not seconds, and route the polling loop
through a background-task mechanism that only re-engages an agent/operator
when the state actually changes -- never a loop where each check consumes a
full attended turn. A 60-second-interval loop that requires active attention
every cycle to babysit a multi-hour job burns disproportionate effort for no
additional information; every experiment in this project that has needed
multi-hour monitoring has used a 5-minute background poll instead, with no
loss of responsiveness once a job actually finishes.

Kill stray polling processes once a job resolves or once monitoring is handed
off to a different mechanism. Orphaned polling loops from an earlier session
or an earlier attempt accumulate silently and are a direct, observed cause of
account-wide rate-limiting on this project (nine stray monitors were found
running simultaneously for already-finished E005/E006 kernels during E007
preparation).

## Session-length ceiling and two-stage runs

Kaggle sessions have a fixed ceiling (12 hours at time of writing). Before
launching, project full-run time from measured per-step throughput on the
same recipe at a smaller scale, and if the projection falls close to that
ceiling, split the run into two kernels ahead of time rather than risking the
whole run: a Phase A kernel that trains to a deliberately chosen intermediate
step and stops, and a Phase B kernel that resumes the exact optimizer,
scheduler, scaler, RNG, and trainer state from Phase A's checkpoint and
continues to the original target. Phase B must not be published until Phase
A's exact checkpoint archive hash has been downloaded and independently
verified -- this is one continuous training run split across two sessions for
infrastructure reasons, not two separate experiments, and it must not restart
from Phase A's adapter as a fresh optimization. E007 uses this pattern.

Uploading the resume checkpoint as a dataset hits a real platform gotcha:
Kaggle's dataset ingestion **recursively auto-extracts any archive it
finds**, confirmed directly -- a `checkpoint-NNNNNN.tar.gz` uploaded as-is
came out on the mounted kernel as `checkpoint-NNNNNN/<tar's own top-level
dir>/*`, individual files, not the intact archive; wrapping it in an outer
zip first did not help, it came out exactly as unpacked. `one_file()`
matching an exact filename will not find an auto-extracted archive at all.
Do not fight this -- verify the checkpoint by hashing each extracted file
individually (computed locally from the same archive already hash-verified
against the source run's own reported per-file hashes) instead of one
whole-archive hash, and locate the checkpoint by its now-nested directory
name instead of the archive's filename. See E007 Phase B's
`install_resume()` in `scripts/training/run_e007_kaggle.py` for the working
pattern.

## A deliberate `enable_internet: true` exception (E011)

Every kernel through E010 sets `enable_internet: false` and pre-stages all
Python dependencies as an offline wheelhouse dataset (the `e003-runtime`
pattern: `pip install --no-index --no-deps --find-links <runtime-dir> ...`).
E011 (the Omnilingual ASR zero-training bake-off) breaks this pattern on
purpose: `omnilingual-asr` depends on `fairseq2`, which ships
platform/CUDA-specific wheels, unlike this project's own pure-PyTorch
runtime bundle -- there is no practical offline wheelhouse to pre-stage for
it, and the model checkpoints themselves are fetched by the library at
first use from Meta's own CDN, not a HuggingFace Hub ID that could be
pre-downloaded and re-uploaded as a Kaggle dataset either. Both the
package and the model card downloads need internet access at kernel
runtime. This is scoped to one exploratory, zero-training evaluation
kernel, not a change to how any training kernel operates.

## `pip install omnilingual-asr` silently downgrades numpy, breaking torch

E011 kernel version 1 crashed on `ValueError: numpy.dtype size changed, may
indicate binary incompatibility. Expected 96 from C header, got 88 from
PyObject`, thrown from deep inside `torch._dynamo` the moment
`omnilingual_asr` was imported (its own import chain pulls in
`fairseq2`, which pulls in `torch.compiler`, which pulls in
`torch._dynamo`). Root cause: `pip install omnilingual-asr`'s own
dependency resolution silently downgraded Kaggle's preinstalled numpy 2.x
to 1.26.4 -- ABI-incompatible with the rest of the preinstalled stack
(torch, etc.), which was built against numpy 2.x. The same install also
logged (but did not itself crash on) further version conflicts --
`huggingface-hub` downgraded below what Kaggle's preinstalled
`transformers` wants, `torch`/`torchvision` mismatched -- worth watching
for if a later step touches either. Fix: force numpy back to the 2.x line
immediately after the omnilingual-asr install
(`pip install --force-reinstall --no-deps "numpy>=2,<3"`), before
importing anything from the package. Not fully explained why fairseq2
pins numpy this way -- looked defensive rather than load-bearing, since
nothing in omnilingual-asr's own code failed on the numpy-1-vs-2 question,
only unrelated downstream torch internals that need numpy 2.x to import
at all.

## Verification discipline

Every downloaded result -- training or evaluation -- must be independently
re-hashed locally and compared against the value the job itself recorded,
not merely trusted because the job reported `exit_code: 0`. Scan the
downloaded log for tracebacks, CUDA memory errors, and non-finite loss values
even on a run that reports success. A job declaring success is a claim, not
proof; every experiment in this project treats it as one until the hashes and
the raw log confirm it independently.
