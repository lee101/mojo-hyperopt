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
Linux 6.8.0-136-generic x86-64 with glibc 2.39. Times are the best of repeated
runs on identical input.

| benchmark | before | after | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 0.99 ms | 0.97 ms | 1.28 ms | 1.33x |
| GMM log-density, 200k x 48 | 501.25 ms | 188.90 ms | 349.81 ms | 1.85x |
| GMM log-density GPU, 200k x 48 | 528.46 ms | 26.45 ms | 507.50 ms | 19.18x |
| quantized GMM log-mass, 100k x 32 | 113.33 ms | 65.04 ms | 218.52 ms | 3.36x |
| log-GMM log-density, 200k x 48 | 567.07 ms | 263.75 ms | 773.12 ms | 2.93x |
| categorical log-density, 5M | 50.58 ms | 50.92 ms | 95.99 ms | 1.88x |
| annealing bounds, 5M centers | 13.31 ms | 13.56 ms | 82.44 ms | 6.08x |
| TPE `fmin`, 200 cheap evaluations | 236.20 ms | 212.53 ms | 656.36 ms | 3.09x |
| anneal `fmin`, 500 cheap evaluations | 485.34 ms | 478.40 ms | 923.16 ms | 1.93x |

"before" and "after" are separate `pixi run bench` runs on the same machine,
one for the previous build and one for the unrolled kernels; the upstream column
is from the "after" run. Rows further down the table than the density kernels
moved only within run-to-run noise. The GPU row was 528 ms in the "before" run
because the device request silently fell back to the CPU on that run; with the
same kernels as before it ran on the device again at 26 ms.

Adaptive Parzen uses NumPy's optimized `argsort` and passes its index buffer
directly to Mojo, avoiding the former scalar merge sort and scratch allocation.
The remaining contiguous sigma and normalization loops use SIMD. Categorical
likelihood builds its small log-probability table once and uses SIMD gathers.
Annealing bounds uses SIMD on every size and a reusable eight-worker pool above
one million centers.

The unquantized Gaussian and log-Gaussian densities walk eight mixture
components per step with eight independent partial sums, and the per-component
normalising constant is hoisted out of the inner loops (the log-Gaussian row
`log(samples[i])` is computed once per sample instead of twice per component).
This changes the floating-point summation order, so results can differ from the
previous build and from Hyperopt in the last bits; the parity tests keep their
original tolerances. Quantized GMM uses the same eight-way accumulation.
Measured effect on the density kernels at the benchmark sizes: GMM 491 ms to
182 ms, log-GMM 566 ms to 264 ms, quantized GMM 113 ms to 64 ms, and every
output agrees with the previous build to within 2e-15.

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
