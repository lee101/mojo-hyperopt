# mojo-hyperopt

`mojo-hyperopt` is a standalone implementation of Hyperopt's TPE and
annealing search algorithms. Python owns search-space traversal, objective
calls, and trial records; Mojo performs the density-estimation kernels that
dominate larger TPE workloads.

The public names follow Hyperopt's common API, so covered programs generally
only change their import:

```python
import numpy as np
import mojohyperopt as hyperopt

space = {
    "x": hyperopt.hp.uniform("x", -5.0, 5.0),
    "scale": hyperopt.hp.loguniform("scale", -3.0, 2.0),
    "model": hyperopt.hp.choice("model", ["linear", "tree"]),
}

trials = hyperopt.Trials()
best = hyperopt.fmin(
    lambda p: (p["x"] - 1.25) ** 2 + (np.log(p["scale"]) + 0.5) ** 2,
    space,
    algo=hyperopt.tpe.suggest,
    max_evals=80,
    trials=trials,
    rstate=np.random.default_rng(7),
    verbose=False,
)
print(best)
print(hyperopt.space_eval(space, best))
print(trials.best_trial["result"]["loss"])
```

Use `hyperopt.anneal.suggest` in place of `hyperopt.tpe.suggest` to run the
annealing search.

## Coverage

The implemented subset is:

| API | Coverage |
| --- | --- |
| search | `tpe.suggest`, configurable startup count, EI candidate count, gamma and prior weight; `anneal.suggest`, `suggest_batch`, `AnnealingAlgo` |
| spaces | recursive dictionaries, lists and tuples; conditional `choice` and `pchoice` branches |
| distributions | `uniform`, `uniformint`, `quniform`, `loguniform`, `qloguniform`, `normal`, `qnormal`, `lognormal`, `qlognormal`, `randint`, `choice`, `pchoice` |
| driver | `fmin`, `space_eval`, `Trials`, `Domain`, status/job constants, scalar or result-dictionary objectives, points to evaluate, early stopping and caught evaluation failures |
| TPE numerics | adaptive Parzen windows, linear forgetting, good/bad trial splitting, Gaussian and log-Gaussian mixture likelihoods, categorical likelihoods |

This is deliberately not the whole Hyperopt project. Arbitrary Pyll
expressions, `Ctrl`, `pass_expr_memo_ctrl`, asynchronous queues,
`SparkTrials`, `MongoTrials`, ATPE, trial persistence, and distributed workers
are not implemented. `max_queue_len` is validated but execution remains
sequential. Search spaces must be built from the `hp` constructors listed
above.

The runtime does not import or delegate to Python Hyperopt. Hyperopt 0.2.7 is
present in the Pixi development environment solely for parity tests and
benchmarks. The tests compare numerical kernels directly with its
implementations and run both optimizers on the same objectives. Hyperopt
0.2.7's annealer requires a one-line scalar `.item()` compatibility correction
under NumPy 2; the tests and benchmark apply that correction to the upstream
reference.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` uses the pinned Mojo nightly and writes
`dist/libmojo-hyperopt.so`. Importing the package also rebuilds a missing or
stale library. For a copied Python installation, set `MOJO_HYPEROPT_LIB` to an
already-built shared library.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-147-generic x86-64 with glibc 2.39. Times are the best of repeated
runs on identical input (one warm-up call, then the fastest of 2-3 timed calls).

**These figures were re-measured in the VERIFY phase.** Every number below is
the median of three full `pixi run bench` runs made in one session, on the
current built library, with the host load average between 27 and 31 on 36
cores. All three runs are in `bench/verify.md`. Both sides of each row run the
same buffers, built from the same seed, and the benchmark compares the two
outputs (`allclose`, `rtol=1e-6, atol=3e-7`) after timing and aborts on a
mismatch, so neither side skips work.

| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 1.02 ms | 1.42 ms | 1.39x |
| GMM log-density, 200k x 48 | 58.14 ms | 426.63 ms | 7.34x |
| GMM log-density GPU, 200k x 48 | 98.94 ms | 406.55 ms | 4.11x |
| quantized GMM log-mass, 100k x 32 | 66.63 ms | 243.93 ms | 3.66x |
| log-GMM log-density, 200k x 48 | 76.02 ms | 545.10 ms | 7.17x |
| categorical log-density, 5M | 35.80 ms | 75.35 ms | 2.11x |
| annealing bounds, 5M centers | 9.67 ms | 48.65 ms | 5.03x |
| TPE `fmin`, 200 cheap evaluations | 212.59 ms | 677.97 ms | 3.19x |
| anneal `fmin`, 500 cheap evaluations | 463.43 ms | 909.44 ms | 1.96x |

The `GMM log-density GPU` row is **not a device measurement**: the wrapper's
4000 MiB free-memory gate declines the device after the first GPU call in a
process, so by the time the row is timed it is the CPU kernel against
upstream's CPU kernel. Its 4.11x is a CPU figure and moves with the GMM row.
The real device comparison is the paired table in the GPU section below.

### SIMD phase

Earlier versions of this table showed a "before (two-pass, scalar inner loop)"
and an "after (SIMD rows)" column side by side. **The "before" column cannot be
reproduced from this tree**: it came from a pre-SIMD library that is no longer
built, and nothing committed reproduces it. Those numbers have been removed
rather than restated. What is reproducible is the current library (the table
above) and the arithmetic of the change, which is still in the source:
`logsumexp_row` forms component terms with `W`-lane loads, in two accumulating
passes (a maximum pass, then an exponential-sum pass with four accumulators), a
scalar tail, with `log(x)` hoisted once per row for the log-normal space.
`tests/test_simd_lengths.py` drives every vectorised kernel at lengths 0, 1,
W-1, W, W+1, 4W, 4W+1, 97 and 100001 for component counts 1, W-1, W, W+1, 3W
and 48, so the scalar tails run. Deleting one tail iteration makes 54 of those
tests fail.

Only the two density kernels changed in the SIMD phase. Its paired
before/after medians (GMM 189.8 to 163.2 ms, log-GMM 276.0 to 228.2 ms) came
from two builds loaded alternately in one process; both builds are now the same
build, so that comparison is no longer reproducible and is not restated as a
current measurement. What this phase's kernels do today is the table above. The
adaptive Parzen, quantized GMM, categorical and annealing-bounds rows do not use
the two density kernels and their row-to-row movement is host noise.

| kernel | what the phase did |
| --- | --- |
| `gmm_lpdf`, `lgmm_lpdf` (unquantized) | Component terms are formed with `W`-lane loads (`component_log_terms`) instead of one scalar division and log per component: a maximum pass and a second exponential-sum pass, with four SIMD accumulators in the sum pass and a scalar tail. `log(x)` for the log-normal space is hoisted to once per row. This is the committed code; confirmed by reading `logsumexp_row` in `src/hyperopt.mojo`. |
| fused single pass over the same data | **Not in the code**; the phase reports it was tried and reverted, citing 200 vs 189.8 ms (GMM) and 272 vs 258 ms (log-GMM). Those two builds no longer exist, so this session could not re-measure the comparison. |
| quantized `gmm_lpdf` line integral | **Not in the code**: `erf` is called from the scalar `normal_cdf`/`lognormal_cdf` only, and every `erf` call site here has `|z| <= 1` (a normalised CDF argument), so the `|x| <= 1` caveat does not apply to the shipped path. The 64.4 vs 63.7 ms revert could not be re-measured. |
| 8 accumulators instead of 4 | **Not in the code**: `logsumexp_row` uses four accumulators. The 178 vs 159 ms revert could not be re-measured. |
| `mho_anneal_bounds` | Single fused SIMD pass over contiguous input with two streamed outputs; measured 9.67 ms at 5M centers this session. |
| `adaptive_parzen`, `categorical_lpdf` | SIMD in the contiguous loops; measured 1.02 ms at 20k and 35.80 ms at 5M this session. |

The SIMD width is read from the built library (`mojohyperopt.tpe.simd_width()`,
exported as `mho_simd_width`), and `tests/test_simd_lengths.py` drives every
vectorised kernel at lengths 0, 1, W-1, W, W+1, 4W, 4W+1, 97 and 100001 for
component counts 1, W-1, W, W+1, 3W and 48, so the scalar tails run. Deleting
one tail iteration makes 54 of those tests fail.

The unquantized density kernels keep the earlier unroll's summation shape:
lanes are accumulated separately and merged once, and the log-Gaussian row
`log(samples[i])` is computed once per sample instead of twice per component.
Adaptive Parzen uses NumPy's optimized `argsort` and passes its index buffer
directly to Mojo; its sigma and normalisation loops, the categorical gather and
the annealing-bounds clamp/offset pass are SIMD over contiguous buffers, with
annealing bounds above one million centers split across a reusable eight-worker
pool. (`acceptance()`, the `low`/`high` normalisation shared by every
GMM/LGMM call, is a scalar loop over `k` components — `k` is bounded by the
trial count and it is not in the hot per-row path.)
Summation order still differs from upstream in the last bits, so the parity
tests keep their `rtol=1e-6, atol=3e-7` tolerances. Re-measured this session
over 50 random kernel comparisons (1 to 4000 samples, 1 to 48 components), the
worst absolute difference from upstream was **1.10e-8**. At the benchmark sizes
it is GMM 1.20e-9, log-GMM 1.71e-8 and quantized GMM 1.74e-8 — all inside that
tolerance, and the largest is 1.7e-8 rather than the 1.3e-8 this section used
to claim.

### Parallel phase

The two unquantized density kernels — the only remaining single-core cost — now
run their rows on up to four threads, but only above a measured size. The
threading is in-kernel `parallelize` from `max.algorithm`; the `std.algorithm`
and `std.threading` forms do not compile on this toolchain.

| benchmark | before (serial rows) | after (threaded rows) | hyperopt 0.2.7 (after run) | speedup |
| --- | ---: | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | — | 1.02 ms | 1.42 ms | 1.39x |
| GMM log-density, 200k x 48 | 158.05 ms | 58.14 ms | 426.63 ms | 7.34x |
| GMM log-density GPU, 200k x 48 | — | 98.94 ms | 406.55 ms | 4.11x |
| quantized GMM log-mass, 100k x 32 | — | 66.63 ms | 243.93 ms | 3.66x |
| log-GMM log-density, 200k x 48 | 229.48 ms | 76.02 ms | 545.10 ms | 7.17x |
| categorical log-density, 5M | — | 35.80 ms | 75.35 ms | 2.11x |
| annealing bounds, 5M centers | — | 9.67 ms | 48.65 ms | 5.03x |
| TPE `fmin`, 200 cheap evaluations | — | 212.59 ms | 677.97 ms | 3.19x |
| anneal `fmin`, 500 cheap evaluations | — | 463.43 ms | 909.44 ms | 1.96x |

The "after" and upstream columns are the same medians as the table above. The
"before (serial rows)" column is re-measured this session by importing the
module, setting `tpe.PARALLEL_WORKERS = 1`, and timing the identical call —
that is the only way the serial path is reachable, since `bench/bench.py` does
not expose the knob. Rows the parallel phase does not touch (`—`) were reported
as unchanged; their serial and threaded paths are the same code, so no separate
serial figure is given here rather than a number that would just be a second
noise sample of the same measurement.

**Five figures in the earlier version of this table did not reproduce within
20%** and are replaced by the measured ones:

| row | previous claim | measured this session |
| --- | --- | --- |
| GMM log-density, 200k x 48, threaded | 55.50 ms | 49.5 — 61.3 ms across runs (median 58.14), 2.6 — 3.2x over serial |
| GMM log-density, 200k x 48, speedup | 9.38x | 6.92 — 7.70x (median 7.34x) |
| log-GMM, 200k x 48, serial rows | 337.47 ms | 229.5 ms |
| quantized GMM, 100k x 32, threaded | 81.42 ms | 66.63 ms |
| quantized GMM, 100k x 32, speedup | 2.97x | 3.66x |

The GMM/log-GMM threaded figures are the least stable in the table: this host
is shared and the load average sat at 27 — 31 on 36 cores during all three
runs, so a 4-thread kernel competes for cores it does not own. The speedup is
real and repeated (serial 158 ms to threaded 49 — 61 ms, bitwise identical
output), but quoting a single best-of-run figure at two decimals overstated it.

Serial against threaded, re-measured directly on the same call this session
(best of 9, `tpe.PARALLEL_WORKERS` toggled between runs):

| kernel, size | serial (workers = 1) | default pool |
| --- | ---: | ---: |
| `anneal_bounds`, 1M centers | 3.98 ms | 2.40 ms |
| `anneal_bounds`, 5M centers | 41.03 ms | 10.67 ms |
| `anneal_bounds`, 20M centers | 202.91 ms | 42.90 ms |

Outputs are `np.array_equal` between the two paths at all three sizes.

`anneal_bounds` was already parallel before this phase (a Python slice pool
above 1M centers) and is left exactly as it was: it is 1.7x faster than a
single serial call at 1M and 3.8 — 4.7x at 5M and 20M, not slower, so there was
nothing to revert. (The earlier table's serial column — 8.23 / 63.53 /
160.16 ms — did not reproduce; these are the measured values, and the 1M row
was noisy enough this session that the pool's advantage there is under 2x, not
the 14x previously implied.) Its 8 workers are the memory-bound case where a
slice split wins; converting it to in-kernel threads was not attempted.

Thread scaling, same kernel, same input, best of 9 (isolated call through the
FFI, `n` samples x `k = 48` components, two independent rounds this session,
load average 31 on a 36-core host):

| k = 48 | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| GMM, n = 8,192 | 6.55 — 7.41 ms | 6.41 — 6.46 ms | 6.39 — 6.47 ms |
| GMM, n = 12,000 | 9.43 — 9.45 ms | 9.52 — 9.59 ms | 9.53 — 9.54 ms |
| GMM, n = 16,384 | 12.80 — 12.99 ms | 12.78 — 13.05 ms | 12.84 — 13.01 ms |
| GMM, n = 32,768 | 25.86 — 25.89 ms | 19.63 — 19.78 ms | 19.44 — 20.27 ms |
| GMM, n = 65,536 | 51.51 — 52.21 ms | 33.89 — 35.34 ms | 21.13 — 23.21 ms |
| GMM, n = 200,000 | 157.47 — 158.40 ms | 87.49 — 88.65 ms | 49.46 — 49.54 ms |
| log-GMM, n = 200,000 | 220.57 — 221.01 ms | 118.85 — 138.29 ms | 62.74 — 74.35 ms |

The crossover at `k = 48` is between 12k and 16k samples: at and below 12k the
threads cost more than they save and the gain is flat (±1%), from 16k it grows
with n. The threshold is therefore **16,384 samples and at least 393,216
sample-component pairs**, then four threads. Which of the two clauses binds
depends on `n` (verified against `mho_parallel_workers` this session):

| `n` | rule | outcome |
| --- | --- | --- |
| < 16,384 or pairs < 393,216 | work gate only | 1 worker |
| 16,384 — 65,535 | work gate, then `min(4, n // 16,384)` | 3 workers at `n = 65535` once `k >= 7`; `n = 16384` stays 1 for any `k` up to 96 |
| >= 65,536 | work gate, then the 4-worker cap | 4 workers as soon as pairs reach 393,216 (at `n = 65,536` that is `k >= 6`, not `k = 6`) |

So the pair clause does real work for `n` between 16,384 and 65,535, and above
`n = 65,536` the size alone gates it. The earlier text claimed the pair clause
kept "a 16k-sample call with 48 components (786k pairs) serial rather than
threaded" while `k = 96` at the same `n` would be threaded; **that does not
reproduce** — at `n = 16384`, `k = 96` also gets 1 worker, because the second
clause caps the count at `n // 16384 = 1`. The width guard is measured at the
sizes where it applies: at `n = 131,072`, `k = 32` takes 74.03 ms serial and
24.28 ms on four threads, against 107.37 ms and 38.11 ms at `k = 48` (the
earlier 74.5 / 31.3 / 104.0 / 40.0 is within 20% of these), so per-row cost
still scales the size gate. 8 threads was re-measured this session and **is
not** a few percent: at `n = 200,000` it is 47.94 ms against 58.25 ms for 4
(1.2x better), and at `n = 65,536` it is 22.42 against 22.82 ms (level). The
earlier text said 8 threads was no better than 4 (31.3 vs 32.9 ms); here it is
modestly better at 200k. The worker cap stays at 4 anyway: the host is
70 — 80% busy during these runs, the gain is under 1.25x, and doubling the
threads taken from other jobs is not justified by it. At 4 threads the speedup
flattens (2.7 — 3.2x at 200k, 2.3 — 2.5x at 65k) while 8 threads adds host
contention, so that flattening is recorded rather than chased. The policy
itself is strict: `mho_parallel_workers` honours requests of 1, 2, 3 or 4
exactly wherever the gates allow, and clamps anything larger to 4 (verified:
request 5 and request 32 both give 4 at `n = 200,000, k = 48`).

How the split works: `parallelize` is handed one work item per **row block** of
`16 x W` rows (128 samples at `W = 8`), not one per row. Blocks are whole
multiples of `W`, so every worker's first SIMD load stays 16-byte aligned, and
the block boundaries — not the worker count — decide which rows go where. The
arithmetic per row is unchanged, so the threaded result is **bitwise identical**
to the serial one; `tests/test_parallel.py` asserts `np.array_equal` (no
tolerance) over both kernels at `n` = 16383, 16384, 16385 and 40001 samples,
`k` = 1, 7 and 48, and 1, 2, 3, 4 and 8 requested workers, plus the exported
`mho_parallel_workers` policy at its boundary sizes.

| change | result |
| --- | --- |
| in-kernel `parallelize` for `gmm_lpdf` / `lgmm_lpdf` rows, threshold 16,384 samples, <= 4 workers | **kept**: 2.6 — 3.2x at 200k x 48 this session (GMM 158.05 → 58.14 ms median, log-GMM 229.48 → 76.02 ms median, bitwise identical output) |
| same path at `n <= 12k` (k = 48), i.e. threading small calls | **not shipped**: 12k is flat (±1%), below it threads lose; the threshold sits above that with margin |
| pair clause of the gate (`n * k >= 393,216`) | **kept, but narrower than documented**: it only binds for 16,384 <= `n` <= 65,535; at `n = 16384` even `k = 96` stays serial (see the rule table above) |
| 8 workers instead of 4 | **not shipped**: measured 47.94 vs 58.25 ms at 200k and 22.42 vs 22.82 ms at 65k — under 1.25x on a 70 — 80%-busy host, not worth the extra cores |
| `anneal_bounds` on the existing 8-thread Python slice pool (1M+ centers) | **kept as-is**: 10.67 ms vs 41.03 ms serial at 5M, 42.90 vs 202.91 ms at 20M (re-measured this session), so it was not reverted |
| any new GPU kernel beyond unquantized GMM/log-GMM | **not shipped**: every other kernel is below 2 flops/byte (CPU only: memory-bound), see the GPU phase below |


### GPU phase

**Verdict: no new GPU path was added, and the existing opt-in one is not
faster at the sizes real callers use.** GMM and log-GMM still accept
`device="gpu"` (CPU remains the default) because that code already shipped and
is correct, but every kernel below the gate is documented as CPU-only and no
caller is routed to the device.

**Gate — flops per byte** (`bench/analysis.md` §5, measured, ~2 flops/byte
threshold):

| kernel | flops/byte | candidate? |
| --- | ---: | --- |
| unquantized `gmm_lpdf` (also its log-space twin) | 6.4 | yes — the only one |
| quantized `gmm_lpdf` | 0.5 | no — CPU only: memory-bound |
| `categorical_lpdf` | 0.1 | no — CPU only: memory-bound |
| `anneal_bounds` | 1.2 | no — CPU only: memory-bound |
| `adaptive_parzen_normal` | 0.16 | no — CPU only: memory-bound |

CPU only: memory-bound for the last four. They stay on the CPU by design, not
because a host API was missing. **The "candidate?" column is an estimate from
an arithmetic flop count, and this session's measurements do not reproduce its
conclusion for the one row it marks "yes".** `gmm_lpdf` at 200k x 48 sustains
only 1.6 — 2.1 Gflop/s on the CPU across `k` = 4 — 192 (measured, flop model:
10 flops per component), against a nominal ~1.15 Tflop/s ceiling. If that were
the whole story, 6.4 flops/byte would put it well above any bandwidth wall: its
minimum traffic is 3.2 MB per call and it runs at ~0.1 GB/s. It is neither
bandwidth-bound nor throughput-bound in the modelled sense, so the >2 flops/byte
rule is not what decides whether the device helps. What actually decides it is
the per-call CUDA context cost, measured below.

**Gate — shared GPU memory.** `nvidia-smi --query-gpu=memory.free` reported
**5101 MiB** free before this session's runs (32.6 GiB card, 12 foreign CUDA
contexts holding ~27.8 GiB), clearing the 4000 MiB floor. Our own device
residency measured **754 MiB** (`nvidia-smi --query-compute-apps`, a new
process appearing with exactly that figure), and free memory fell **5101 MiB →
3548 MiB** after the first device call, which puts later calls below the floor
and makes the product's own gate decline them. That is the 4309 MiB / 3240 MiB
picture the phase reported, re-measured: the host's free figure moves by
hundreds of MiB as other jobs allocate.

**CPU vs `device="gpu"`, re-measured this session.** This is the table that did
not reproduce. Each `n` was run in a **fresh process** with the gate bypassed
(`tpe._gpu_memory_available` stubbed to `True`) so the device was actually
used; CPU is the best of 7, the device column reports both the first call in the
process and the best of 5 after it, K = 48:

| n | CPU | device, first call | device, warm | CPU / warm device | device output vs CPU |
| ---: | ---: | ---: | ---: | ---: | --- |
| 24 (real TPE caller size) | 0.06 ms | 395.33 ms | 2.31 ms | **0.02x** | 4.1e-11 |
| 1,000 | 0.83 ms | 449.16 ms | 2.38 ms | **0.35x** | 7.2e-11 |
| 20,000 | 15.71 ms | 274.46 ms | 0.36 ms | **43.52x** | 7.2e-11 |
| 200,000 (benchmark size) | 57.43 ms | 389.58 ms | 1.42 ms | **40.31x** | 7.2e-11 |
| 1,000,000 | 218.01 ms | 352.75 ms | 10.37 ms | **21.03x** | 7.2e-11 |

The previous version of this table reported `device="gpu"` at 30.55 — 254.10 ms
and a ratio of 0.01 — 0.96x, i.e. "the GPU loses at every size, including the
200k benchmark row". **That does not reproduce.** The warm device call at
n = 200,000 is **1.42 ms against 57.43 ms on the CPU — the device wins by about
40x** (the row-to-row spread is 40 — 43x at 20k and 200k, 21x at 1M in the same
units). The correction also removes the stated cause: a `DeviceContext()` is
built per call, but it does **not** cost ~30 ms per call. The first device call
in a process costs **274 — 449 ms**, and that is a one-time CUDA context and
module-load cost (the `nvidia-smi` residency jump confirms a fresh context is
being created). Every later call costs about **2 ms of fixed overhead plus the
transfers**, which is why the device loses at n = 24 (0.06 ms of CPU work) and
wins once the kernel is larger than that overhead (between n = 1,000 and
n = 20,000 here).

So the honest reading, with nothing removed, is:

* **A process's first device call is always a loss** (274 — 449 ms against
  0.06 — 218 ms of CPU work): the context creation dominates.
* **On warm calls the device wins from roughly n = 20,000 upward** (40x at the
  200k benchmark size), and loses at the sizes TPE actually calls it with.
* The old claim that "1.47 ms isolated still did not beat a 48.63 ms CPU call
  once the wrapper's 30 ms is added" is contradicted by the measurement: the
  isolated kernel figure agrees (1.4 ms warm here), but there is no 30 ms
  per-call wrapper cost to add. The old verdict therefore rested on an error in
  the per-call cost model, not on the device being slow.

The `device="gpu"` path was not changed and was not reverted: it is opt-in,
it is correct (differences from the CPU path are ≤ 1.9e-9 at every boundary
size measured, see below), and it costs CPU callers nothing. What is now
corrected is the record of what it costs and when it wins.

Real caller sizes are still the point: a 200-evaluation TPE run issues 360
`GMM1_lpdf` calls and **every one is n = 24** (re-measured by wrapping the
wrapper; the counter printed `{24: 360}`). At n = 24 the device is ~40x
*slower*, so the shipped code's choice of CPU as the default is right for the
callers this package has. The 200k benchmark row is synthetic; no caller in
this package passes it.

**Transfer pattern of what ships** (unchanged, and already conforming): one
`DeviceContext`, one `enqueue_create_buffer` per array, inputs copied in once,
**one** kernel launch for the whole input (`grid_dim = ceil(n / 256)`), results
copied out once, then `synchronize`. The kernel returns early for
`i >= n`, so the grid tail is explicit rather than padded. Re-measured this
session against the CPU kernel at every block boundary: maximum absolute
difference **7.2e-11** for GMM at n = 1 / 255 / 256 / 257 / 4097 / 10003 /
200003 and **≤ 1.9e-9** for log-GMM at the same sizes (GMM agrees to 3.9e-11 at
n = 1). The earlier "0.00e+00 at all of them" was wrong: the two paths do not
agree exactly, they agree to about 1e-10.

**GPU tests** use `device="gpu"` and `pytest.skip("GPU unavailable or has less
than 4000 MiB free")` when the device or its memory floor is absent, so a
CPU-only host skips them cleanly. `tests/test_kernels.py` covers both kernels
end to end at n = 10,003 (not a multiple of the 256-thread block), the grid
tail at n = 1/255/256/257/200,003, the 2 GiB budget boundary, the explicit CPU
fallback when the gate says no, and the rejection of an unknown device. This
session ran the suite with **354 passed, 6 skipped**; all six skips are the GPU
cases carrying the reason string above.

On this shared host, **a process gets one GPU call and later calls skip, for a
real reason**: the first device call allocates a 754 MiB context that stays
resident, so `nvidia-smi` free memory goes from **5101 MiB to 3548 MiB** and
the product's own 4000 MiB floor then declines every later call in that
process. Measured directly this session: `_gpu_memory_available` is `True` for
n = 1/255/256/257/200,003 before any call and `False` for all of them after one
`device="gpu"` call, with free memory reading 4309 MiB before and 3550 MiB
after. Which device cases land before that first allocation depends on whether
the card is above 4000 MiB at that moment (another job on this box holds
~27.8 GiB and its free figure moves by hundreds of MiB). No test fails — every
skip carries the reason string above. On a CPU-only host the same cases skip
for missing hardware.

The `GMM log-density GPU, 200k x 48` benchmark row measures
`GMM1_lpdf(..., device="gpu")` and falls back to the CPU when the gate declines,
so that row is not a device measurement on a contended host; the paired table
above is the device measurement. `bench/bench.py` warms up with one untimed
`ours()` call before timing, and that warm-up is the call that consumes the
process's single device allocation, so the timed repetitions are always CPU
ones. The row is kept in the table because it is what `pixi run bench`
produces, and its numbers are labelled accordingly.

Run `pixi run bench` to reproduce the table. The task takes a machine-wide
file lock so concurrent benchmark jobs do not overlap.

## How it works

TPE keeps Hyperopt's split rule: the best
`min(ceil(gamma * sqrt(n_trials)), 25)` trials form `l(x)` and the remainder
form `g(x)`. It fits adaptive one-dimensional Parzen mixtures to every active
parameter, draws candidates from `l(x)`, evaluates
`log l(x) - log g(x)`, and returns the highest-scoring recursive candidate.
Categorical variables use weighted pseudocounts; conditional branch parameters
only receive observations from trials in which they were active.

Annealing samples a trial rank from the upstream geometric rule and shrinks
each proposal neighborhood by `1 / (1 + observations * shrink_coef)`. Reusing
the chosen trial ID across active parameters preserves correlation within a
conditional point.

The bridge converts NumPy arrays to C-contiguous `float64` or `int64`, rejects
unsafe lengths, shapes, categories and numeric values, and keeps every input,
output and sorting index buffer alive for the synchronous call. ctypes passes
each buffer as an integer address, and the exported Mojo functions validate
lengths and non-null addresses before reconstructing
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the C ABI boundary. There are
no Mojo-side allocations on the CPU path and one shared-library call processes
a complete batch. The optional GPU path creates bounded device buffers, copies
the existing contiguous arrays, and releases those buffers when the call
returns.

The random number generator remains NumPy's `Generator`, preserving seeded
reproducibility within this implementation. Candidate streams need not be
bit-for-bit identical to upstream because mixture draws are batched
differently; density values and optimizer behavior are parity-tested instead.

## License

MIT
