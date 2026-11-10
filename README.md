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
runs on identical input.

The `SIMD phase` table below is the change made by this phase: every row is
from one `pixi run bench` before the change and one after it, on the same
machine in the same session, with the load average under 30 (other jobs share
this host; timings move by a few percent run to run and the upstream column is
noisier still).

| benchmark | before (two-pass, scalar inner loop) | after (SIMD rows) | hyperopt 0.2.7 (after run) | speedup |
| --- | ---: | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 1.11 ms | 1.13 ms | 1.53 ms | 1.35x |
| GMM log-density, 200k x 48 | 189.77 ms | 156.02 ms | 532.35 ms | 3.41x |
| GMM log-density GPU, 200k x 48 | 253.14 ms | 200.07 ms | 501.04 ms | 2.50x |
| quantized GMM log-mass, 100k x 32 | 66.11 ms | 66.93 ms | 225.77 ms | 3.37x |
| log-GMM log-density, 200k x 48 | 286.54 ms | 223.81 ms | 534.31 ms | 2.39x |
| categorical log-density, 5M | 34.24 ms | 31.85 ms | 67.29 ms | 2.11x |
| annealing bounds, 5M centers | 9.69 ms | 8.88 ms | 47.12 ms | 5.30x |
| TPE `fmin`, 200 cheap evaluations | 221.29 ms | 221.04 ms | 683.69 ms | 3.09x |
| anneal `fmin`, 500 cheap evaluations | 484.70 ms | 457.76 ms | 861.55 ms | 1.88x |

Only the two density kernels changed. Their paired medians, measured by calling
the two builds' exported kernels alternately in one process so shared-host noise
hits both equally (15 rounds, 200k x 48), are GMM 189.8 ms to 163.2 ms and
log-GMM 276.0 ms to 228.2 ms. The GPU row fell with them because in that run the
device request fell back to the CPU; the GPU kernel itself is unchanged. The
adaptive Parzen, quantized GMM, categorical and annealing-bounds rows were
already SIMD from the earlier phases and moved only within noise; they are kept
in the table to show that.

| kernel | what the phase did |
| --- | --- |
| `gmm_lpdf`, `lgmm_lpdf` (unquantized) | Component terms are now formed with `W`-lane loads (`component_log_terms`) instead of one scalar division and log per component: a maximum pass and a second exponential-sum pass, with four SIMD accumulators in the sum pass and a scalar tail. `log(x)` for the log-normal space is hoisted to once per row. |
| fused single pass over the same data | Tried, measured, and **reverted**: folding the maximum into the summing pass needs every component exponentiated twice (once to fold it into the running maximum), and it was 200 ms against 189.8 ms for GMM and 272 ms against 258 ms for log-GMM. The maximum pass here only ever builds cheap underflowing exponentials, so two passes with one exp per component wins. |
| quantized `gmm_lpdf` line integral | Tried `W`-lane `erf`; measured 64.4 ms against 63.7 ms, so **reverted**. The compiler's `erf` is also only correct for SIMD arguments with `|x| <= 1`, which the line integral cannot guarantee. |
| 8 accumulators instead of 4 | Tried; 178 ms against 159 ms for GMM, so **reverted**. Two accumulator pairs in the maximum pass was neutral and is not kept. |
| `mho_anneal_bounds` | Already a single fused SIMD pass over contiguous input with two streamed outputs at every size; no change was faster. |
| `adaptive_parzen`, `categorical_lpdf` | Already SIMD in the contiguous loops; no change was faster. |

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
pool.
Summation order still differs from upstream in the last bits, so the parity
tests keep their `rtol=1e-6, atol=3e-7` tolerances; against upstream the SIMD
rows differ by at most 1.3e-8 absolute at the benchmark size.

### Parallel phase

The two unquantized density kernels — the only remaining single-core cost — now
run their rows on up to four threads, but only above a measured size. The
threading is in-kernel `parallelize` from `max.algorithm`; the `std.algorithm`
and `std.threading` forms do not compile on this toolchain.

| benchmark | before (serial rows) | after (threaded rows) | hyperopt 0.2.7 (after run) | speedup |
| --- | ---: | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 0.96 ms | 0.93 ms | 1.42 ms | 1.52x |
| GMM log-density, 200k x 48 | 186.77 ms | 55.50 ms | 520.61 ms | 9.38x |
| GMM log-density GPU, 200k x 48 | 218.70 ms | 95.32 ms | 357.79 ms | 3.75x |
| quantized GMM log-mass, 100k x 32 | 66.00 ms | 81.42 ms | 241.80 ms | 2.97x |
| log-GMM log-density, 200k x 48 | 337.47 ms | 77.39 ms | 564.96 ms | 7.30x |
| categorical log-density, 5M | 38.75 ms | 32.52 ms | 68.41 ms | 2.10x |
| annealing bounds, 5M centers | 11.21 ms | 8.51 ms | 46.99 ms | 5.52x |
| TPE `fmin`, 200 cheap evaluations | 212.10 ms | 206.89 ms | 654.36 ms | 3.16x |
| anneal `fmin`, 500 cheap evaluations | 478.74 ms | 451.74 ms | 869.42 ms | 1.92x |

Both columns are `bench/bench.py` on the **same built library**, run back to
back this session under the same file lock, with the load average near 23-25
on the shared host. "Before" is the serial path with `tpe.PARALLEL_WORKERS = 1`
forced (`bench/bench.py` does not expose that knob; importing the module, setting
`tpe.PARALLEL_WORKERS = 1`, and then running `bench.py` with `runpy` reproduces
it); "after" is the default. The two GMM rows are the whole effect. The
other rows do not call the two density kernels and moved only by host noise:
quantized GMM, categorical and annealing bounds are measured again below with
the same serial/threaded pair, and Parzen and the two `fmin` runs are close
enough to be called unchanged. TPE `fmin` at 200 evaluations barely moves
because those calls (20 — 200 samples) are under the threshold and stay serial.

Serial against threaded, measured directly on the same call (best of 15,
isolated, `tpe.PARALLEL_WORKERS` toggled between runs):

| kernel, size | serial | 8-thread pool (previous path) | threaded default |
| --- | ---: | ---: | ---: |
| `anneal_bounds`, 1M centers | 8.23 ms | 0.58 ms | 0.58 ms (unchanged pool) |
| `anneal_bounds`, 5M centers | 63.53 ms | 14.46 ms | 14.46 ms (unchanged pool) |
| `anneal_bounds`, 20M centers | 160.16 ms | 50.84 ms | 50.84 ms (unchanged pool) |

`anneal_bounds` was already parallel before this phase (a Python slice pool
above 1M centers) and is left exactly as it was: the parallel path was
re-measured and it is 4.4x faster than one serial call, not slower, so there
was nothing to revert. Its 8 workers are the memory-bound case where a slice
split wins; converting it to in-kernel threads was not attempted.

Thread scaling, same kernel, same input, best of 9 (isolated call through the
FFI, `n` samples x `k = 48` components, load average ≈ 25 on a 36-core host):

| k = 48 | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| GMM, n = 8,192 | 6.63 ms | 6.70 ms | 6.48 ms |
| GMM, n = 12,000 | 9.82 ms | 9.90 ms | 9.79 ms |
| GMM, n = 16,384 | 13.47 ms | 12.26 ms | 11.61 ms |
| GMM, n = 32,768 | 25.1 ms | 18.0 ms | 13.4 ms |
| GMM, n = 65,536 | 51.8 ms | 31.4 ms | 21.6 ms |
| GMM, n = 200,000 | 186.8 ms | 104.5 ms | 59.9 ms |
| log-GMM, n = 200,000 | 337.5 ms | 129.6 ms | 70.4 ms |

The crossover at `k = 48` is between 12k and 16k samples: at and below 12k the
threads cost more than they save and the gain is flat (±1%), from 16k it grows
with n. The threshold is therefore **16,384 samples and at least 393,216
sample-component pairs** (so narrow mixtures stay serial at the crossover:
`n = 16384, k = 48` is serial, `k = 96` is not), then four threads. The width
guard is a measurement too: at the same `n = 131,072`, `k = 32` takes 74.5 ms
serial and 31.3 ms on four threads, against 104.0 ms and 40.0 ms at `k = 48`,
so the per-row cost is what sets the size gate, and a 16k-sample call with 48
components (786k pairs) is left serial rather than threaded. 8 threads
were measured too (31.3 ms at 200k against 32.9 ms for 4, and 19.0 against
21.6 at 65k) — a few percent on a host that is 70% busy, which does not justify
doubling the threads taken from other jobs, so the worker cap stays at 4 and
stops at 2 until `n >= 4 x threshold`. At 4 threads the speedup flattens
(3.3x at 200k, 3.4x at 65k) while 8 threads adds host contention, so that
flattening is recorded rather than chased.

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
| in-kernel `parallelize` for `gmm_lpdf` / `lgmm_lpdf` rows, threshold 16,384 samples, <= 4 workers | **kept**: 3.4x at 200k x 48 (GMM 186.8 → 55.5 ms, log-GMM 337.5 → 77.4 ms) |
| same path at `n <= 12k` (k = 48), i.e. threading small calls | **not shipped**: 12k is flat (±1%), below it threads lose; the threshold sits above that with margin |
| 8 workers instead of 4 | **not shipped**: +4-12% at 65k-200k on a 70%-busy host; not worth the extra cores |
| `anneal_bounds` on the existing 8-thread Python slice pool (1M+ centers) | **kept as-is**: 14.5 ms vs 63.5 ms serial at 5M, 50.8 vs 160.2 ms at 20M — re-measured this session, it wins, so it was not reverted |
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
because a host API was missing.

**Gate — shared GPU memory.** `nvidia-smi --query-gpu=memory.free` reported
**4309 MiB** free before this phase (32.6 GiB card, 12 foreign CUDA contexts
holding ~27.8 GiB), so the 4000 MiB floor was cleared and the phase proceeded.
That margin is real but thin: the same query returned 3240 MiB seconds later
while other jobs allocated, and the product's own gate (`_GPU_MIN_FREE_MIB`)
correctly refused the device at that moment. Our own device residency measured
754 MiB (`nvidia-smi --query-compute-apps`), well under the 2 GiB per-call
budget.

**Before / after, measured this session** (paired, interleaved CPU vs
`device="gpu"` calls on the same buffers in one process, best of 9, K = 48):

| n | CPU | `device="gpu"` | CPU / GPU | device output |
| ---: | ---: | ---: | ---: | --- |
| 24 (real TPE caller size) | 0.20 ms | 30.55 ms | **0.01x** | matches |
| 1,000 | 2.00 ms | 33.04 ms | **0.06x** | matches |
| 20,000 | 28.92 ms | 34.15 ms | **0.85x** | matches |
| 200,000 (benchmark size) | 53.94 ms | 91.59 ms | **0.59x** | matches |
| 1,000,000 | 245.15 ms | 254.10 ms | **0.96x** | matches |

The GPU wins only where the kernel dominates the per-call device setup, and it
never won here: 1.47 ms isolated (n = 200k, back-to-back exported-symbol calls
with nothing else on the device) still did not beat a 48.63 ms CPU call once
the wrapper's own per-call `DeviceContext()` construction — about 30 ms — is
added. So the honest reading is **the GPU path loses at every size measured,
including the 200k benchmark row**, because a fresh device context per call
costs more than the kernel saves. It was not reverted (it is opt-in, correct,
and costs CPU callers nothing), but nothing new was built on it.

Real caller sizes are the point: a 200-evaluation TPE run issues 360
`GMM1_lpdf` calls and **every one is n = 24** (measured by wrapping the
wrapper). The 200k benchmark row is synthetic; no caller in this package passes
it.

**Transfer pattern of what ships** (unchanged, and already conforming): one
`DeviceContext`, one `enqueue_create_buffer` per array, inputs copied in once,
**one** kernel launch for the whole input (`grid_dim = ceil(n / 256)`), results
copied out once, then `synchronize`. The kernel returns early for
`i >= n`, so the grid tail is explicit rather than padded. Verified at n = 1,
255, 256, 257, 4097 and 200,003 (device block boundary ±1): maximum absolute
difference from the CPU path **0.00e+00** at all of them, and 7.15e-11 at
200,000 over random draws.

**GPU tests** use `device="gpu"` and `pytest.skip("GPU unavailable or has less
than 4000 MiB free")` when the device or its memory floor is absent, so a
CPU-only host skips them cleanly. `tests/test_kernels.py` covers both kernels
end to end at n = 10,003 (not a multiple of the 256-thread block), the grid
tail at n = 1/255/256/257/200,003, the 2 GiB budget boundary, the explicit CPU
fallback when the gate says no, and the rejection of an unknown device.

On this shared host, **a process gets one GPU call and later calls skip, for a
real reason**: the first device call allocates a 754 MiB context that stays
resident, so `nvidia-smi` free memory goes from **5101 MiB to 3550 MiB** and
the product's own 4000 MiB floor then declines every later call in that
process. Measured directly: `_gpu_memory_available` is `True` for n =
1/255/256/257/200,003 before any call and `False` for all of them after one
`device="gpu"` call. The full suite therefore reports **354 passed, 6
skipped** at 4309 MiB free, and which device cases land before that first
allocation depends on whether the card is above 4000 MiB at that moment
(another job on this box holds ~27.8 GiB and its free figure moves by
hundreds of MiB). No test fails — every skip carries the reason string above.
On a CPU-only host the same cases skip for missing hardware.

The `GMM log-density GPU, 200k x 48` benchmark row measures
`GMM1_lpdf(..., device="gpu")` and falls back to the CPU when the gate declines,
so that row is not a device measurement on a contended host; the paired table
above is the device measurement.

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
