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

Unbounded, unquantized GMM and log-GMM evaluation also accept `device="gpu"`.
This is an explicit opt-in because transfers and device setup are not worthwhile
for small inputs; CPU remains the default. The GPU path checks that at least
4000 MiB is free, refuses requests requiring 2 GiB or more of device buffers,
and falls back to CPU if no suitable device is available. Once a GPU call is
attempted, a kernel or transfer failure is raised rather than hidden. The GPU
benchmark above required about 3.1 MiB of device buffers for the 200k-sample
workload.

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
