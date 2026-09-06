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

## Verification discipline

Every downloaded result -- training or evaluation -- must be independently
re-hashed locally and compared against the value the job itself recorded,
not merely trusted because the job reported `exit_code: 0`. Scan the
downloaded log for tracebacks, CUDA memory errors, and non-finite loss values
even on a run that reports success. A job declaring success is a claim, not
proof; every experiment in this project treats it as one until the hashes and
the raw log confirm it independently.
