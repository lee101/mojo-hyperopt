# VERIFY: reproduction of the mojo-hyperopt benchmark claims

Phase: VERIFY. Written by the verifying pass, not by the code's author.

Everything below was run in this session on this machine, on the library built
by `pixi run build` from commit `66063b7` (working tree clean apart from the
documentation edits listed at the end). No source file under `src/` or `python/`
was changed, and no test was changed.

| item | value |
| --- | --- |
| CPU | Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs |
| OS / libc | Linux 6.8.0-147-generic x86_64, glibc 2.39 |
| GPU | ~32.6 GiB card shared with 12 foreign CUDA contexts (~27.8 GiB held) |
| host load average during runs | 27 — 31 (36 physical cores; other jobs share the box) |
| Mojo / max | 1.2.0.dev2026092605 / 26.7.0.dev2026092605 (from `pixi.toml`) |
| Python | 3.13 (pixi env) |
| upstream | hyperopt 0.2.7 from the pixi env, NumPy-2 `.item()` shim as in `bench.py` |

## 1. Gate: build and test

```
pixi run build   -> exit 0
pixi run test    -> 354 passed, 6 skipped in 7.14s   (exit 0)
```

The 6 skips are the GPU cases, each carrying
`"GPU unavailable or has less than 4000 MiB free"`. They skip because the card
fell below the product's 4000 MiB floor after the first device context in the
suite's process, not because of missing hardware — see §6.

The suite was run again at the very end of this phase with the same result
(§9).

## 2. `pixi run bench`, three full runs

The task is `flock /tmp/mojo-bench.lock python bench/bench.py`. Run as-is, no
modification, one after another. Each run is the file's own protocol: one
untimed warm-up per row, then the fastest of 2 — 3 timed repetitions.

Run 1 (load average 29.57):

| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 1.02 ms | 1.42 ms | 1.39x |
| GMM log-density, 200k x 48 | 61.61 ms | 426.63 ms | 6.92x |
| GMM log-density GPU, 200k x 48 | 93.97 ms | 424.79 ms | 4.52x |
| quantized GMM log-mass, 100k x 32 | 65.64 ms | 233.48 ms | 3.56x |
| log-GMM log-density, 200k x 48 | 76.02 ms | 557.77 ms | 7.34x |
| categorical log-density, 5M | 35.80 ms | 75.50 ms | 2.11x |
| annealing bounds, 5M centers | 9.67 ms | 48.78 ms | 5.05x |
| TPE `fmin`, 200 cheap evaluations | 253.42 ms | 712.79 ms | 2.81x |
| anneal `fmin`, 500 cheap evaluations | 482.00 ms | 947.64 ms | 1.96x |

Run 2 (load average 28.23):

| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 1.17 ms | 1.56 ms | 1.33x |
| GMM log-density, 200k x 48 | 58.14 ms | 447.86 ms | 7.70x |
| GMM log-density GPU, 200k x 48 | 100.72 ms | 353.02 ms | 3.50x |
| quantized GMM log-mass, 100k x 32 | 69.43 ms | 243.93 ms | 3.51x |
| log-GMM log-density, 200k x 48 | 75.43 ms | 545.10 ms | 7.23x |
| categorical log-density, 5M | 33.88 ms | 66.78 ms | 1.97x |
| annealing bounds, 5M centers | 8.45 ms | 47.63 ms | 5.64x |
| TPE `fmin`, 200 cheap evaluations | 212.59 ms | 677.65 ms | 3.19x |
| anneal `fmin`, 500 cheap evaluations | 451.04 ms | 884.68 ms | 1.96x |

Run 3 (load average 27.10):

| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 0.97 ms | 1.34 ms | 1.39x |
| GMM log-density, 200k x 48 | 57.00 ms | 395.03 ms | 6.93x |
| GMM log-density GPU, 200k x 48 | 98.94 ms | 406.55 ms | 4.11x |
| quantized GMM log-mass, 100k x 32 | 66.63 ms | 248.83 ms | 3.73x |
| log-GMM log-density, 200k x 48 | 79.39 ms | 538.53 ms | 6.78x |
| categorical log-density, 5M | 37.32 ms | 75.35 ms | 2.02x |
| annealing bounds, 5M centers | 9.90 ms | 48.65 ms | 4.92x |
| TPE `fmin`, 200 cheap evaluations | 208.60 ms | 677.97 ms | 3.25x |
| anneal `fmin`, 500 cheap evaluations | 463.43 ms | 909.44 ms | 1.96x |

Medians (what the README now shows):

| benchmark | mojo (median) | upstream (median) | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k | 1.02 ms | 1.42 ms | 1.39x |
| GMM, 200k x 48 | 58.14 ms | 426.63 ms | 7.34x |
| GMM GPU row, 200k x 48 | 98.94 ms | 406.55 ms | 4.11x |
| quantized GMM, 100k x 32 | 66.63 ms | 243.93 ms | 3.66x |
| log-GMM, 200k x 48 | 76.02 ms | 545.10 ms | 7.17x |
| categorical, 5M | 35.80 ms | 75.35 ms | 2.11x |
| annealing bounds, 5M | 9.67 ms | 48.65 ms | 5.03x |
| TPE `fmin`, 200 evals | 212.59 ms | 677.97 ms | 3.19x |
| anneal `fmin`, 500 evals | 463.43 ms | 909.44 ms | 1.96x |

## 3. Which README rows did not reproduce within 20%

Judged against the figures the README carried before this phase. "miss" means
the measured value is outside ±20% of the claimed value.

| row / column | claimed | measured this session | verdict |
| --- | ---: | ---: | --- |
| GMM 200k x 48, threaded (parallel table) | 55.50 ms | 49.46 — 61.28 ms, median 58.14 | within 20% by worst case; speedup miss |
| GMM 200k x 48, speedup (parallel table) | 9.38x | 6.92 — 7.70x | **miss** (−18% to −26%) |
| log-GMM 200k x 48, serial rows | 337.47 ms | 229.48 ms | **miss** (−32%) |
| quantized GMM 100k x 32, threaded | 81.42 ms | 66.63 ms | **miss** (−18%) |
| quantized GMM 100k x 32, speedup | 2.97x | 3.51 — 3.73x | **miss** (+18% to +26%) |
| TPE `fmin`, 200 evals, ours | 206.89 ms | 208.60 — 253.42 ms | within 20% (one run is +22%, the other two +1%) |
| GMM, n = 32,768, 4 threads (thread table) | 13.4 ms | 19.44 — 20.27 ms | **miss** (+45%) |
| `anneal_bounds`, 5M, threaded | 14.46 ms | 10.67 ms | **miss** (−26%) |
| `anneal_bounds`, 5M, serial | 63.53 ms | 41.03 ms | **miss** (−35%) |
| `anneal_bounds`, 1M, pool vs serial | 0.58 vs 8.23 ms (14x) | 2.40 vs 3.98 ms (1.7x) | **miss** |
| `anneal_bounds`, 20M, pool | 50.84 ms | 42.90 ms | within 20% (−16%) |
| 8-thread vs 4-thread at 200k | no gain (31.3 vs 32.9) | 47.94 vs 58.25 ms | **miss** (sign of direction) |
| SIMD "before (two-pass, scalar inner loop)" column | 9 values | not reproducible | **unverifiable** (§4) |
| SIMD "after" column + upstream + speedups | 9 rows | reproduced within 20% except GMM speedup 3.41x vs 6.92 — 7.70x measured | **miss** (that column is a different, older build) |
| GPU `device="gpu"` column | 30.55 — 254.10 ms | see §6 | **miss** (by up to 70x) |
| GPU parity at block boundaries | "0.00e+00 at all of them" | 7.2e-11 (GMM), ≤ 1.9e-9 (log-GMM) | **miss** (not exact) |
| worst kernel difference from upstream | "at most 1.3e-8" | 1.71e-8 (log-GMM 200k), 1.74e-8 (quantized) | **miss** (mild) |
| `n = 16384, k = 96` is threaded | threaded | 1 worker | **miss** (§7) |

Rows where the port was faster stayed faster; no row is removed or softened.
The lowest row in the table is `anneal fmin` at 1.96x, unchanged in spirit from
the 1.88 — 1.92x the README claimed.

## 4. What could not be reproduced

* **The SIMD phase's "before" column.** It reports a pre-SIMD library
  (two-pass scalar inner loop). That build is not in the tree and no committed
  revision produces it, so the nine "before" figures cannot be re-measured.
  They were removed from the README rather than restated; the "after" column
  was replaced by the current measured medians.
* **The revert measurements in the SIMD "what the phase did" table**
  (`200 vs 189.8 ms`, `64.4 vs 63.7 ms`, `178 vs 159 ms`). These compare
  variants that no longer exist. The README now states what the committed code
  does (verified by reading `logsumexp_row` and `normal_cdf`/`lognormal_cdf`)
  and marks those comparisons as not re-measurable in this session.
* **The GPU phase's paired table as written.** Reproduced, but the numbers are
  different by up to 70x; see §6. This is the most material correction.
* **`bench/analysis.md` §1's `GMM log-density GPU` row (30.05 ms, 17.33x).**
  That row was the CPU fallback, not the device. Annotated in place.

## 5. Is the comparison fair? Same input, no skipped work

All of the following were checked by reading the code and by measurement.

* **Same input on both sides.** Each `bench.py` case builds one set of arrays
  from a fixed seed and hands the *same* arrays to both implementations. The
  `fmin` rows build an identical space and objective on each side and use the
  same `rstate=np.random.default_rng(7)` and the same `max_evals`.
* **Outputs actually compared.** After timing, `bench.py` runs
  `np.allclose(ours, upstream, rtol=1e-6, atol=3e-7)` on every array row and
  raises `AssertionError` on mismatch. The rows ran without raising. The `fmin`
  rows return a scalar loss and are not compared (see the next point).
* **Upstream does the full work.** `hyperopt.tpe.GMM1_lpdf` builds the whole
  `samples[:, None] - mus` matrix; `LGMM1_lpdf` likewise; the quantized branch
  builds the per-component `prob` array for every sample. No early return, no
  sampling, no reduced precision at these sizes.
* **Our wrapper does the full work.** Comparing the public call against a raw
  call of the exported `mho_gmm1_lpdf` symbol with pre-made contiguous buffers,
  quantized 100k x 32: full call 61.21 ms, raw kernel 65.53 ms, outputs
  `np.array_equal` (max diff 0.0). The wrapper is not hiding or deferring work;
  the raw-vs-full difference is noise.
* **Upstream's anneal shim is not a shortcut.** `bench.py`'s
  `NumPy2AnnealingAlgo.choose_ltv` is upstream's own method with
  `np.asarray(...).item()` applied to the NumPy-2 integer that otherwise raises
  `IndexError`. Same algorithm, same parameters (`avg_best_idx=2.0`,
  `shrink_coef=0.1`), same trial count. Both sides run 500 evaluations.
* **Same 200k x 48 shape on both sides** for the density rows; the GPU row
  passes the same arrays to our wrapper with `device="gpu"` and to upstream's
  CPU function — which is why that row is not a like-for-like device
  comparison, and is now labelled as such (§6).
* **Kernel agreement, random sweep.** 50 random comparisons (n = 1 — 4000,
  k = 1 — 48, random weights/sigmas, both kernels, bounded and unbounded):
  worst absolute difference **1.10e-8**. At benchmark sizes: GMM 200k x 48
  **1.20e-9**, log-GMM 200k x 48 **1.71e-8**, quantized 100k x 32 **1.74e-8**.

## 6. GPU: the row that did not reproduce, and why

The wrapper's gate (`_gpu_memory_available`) declines the device whenever
`nvidia-smi` reports under 4000 MiB free. The first `device="gpu"` call in a
process creates a CUDA context that stays resident (754 MiB, confirmed as a new
`--query-compute-apps` entry), and free memory went **5101 MiB → 3548 MiB**.
So the first call in a process may use the device and every later one falls
back to the CPU. `bench.py` warms each row up with one untimed `ours()` call,
which consumes that single allocation; the timed repetitions of the GPU row are
therefore CPU.

To measure the device itself, each size was run in a **fresh process** with
`tpe._gpu_memory_available` stubbed to `True` (so the real kernel executed),
CPU best of 7, device best of 5 after one untimed call, k = 48:

| n | CPU | device, first call | device, warm | CPU / warm device | device vs CPU |
| ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 0.06 ms | 395.33 ms | 2.31 ms | 0.02x | 4.1e-11 |
| 1,000 | 0.83 ms | 449.16 ms | 2.38 ms | 0.35x | 7.2e-11 |
| 20,000 | 15.71 ms | 274.46 ms | 0.36 ms | **43.52x** | 7.2e-11 |
| 200,000 | 57.43 ms | 389.58 ms | 1.42 ms | **40.31x** | 7.2e-11 |
| 1,000,000 | 218.01 ms | 352.75 ms | 10.37 ms | **21.03x** | 7.2e-11 |

Reading:

* The README's claim that `device="gpu"` loses at every size (0.01 — 0.96x,
  30.55 — 254.10 ms) does not hold for warm calls: at 200k the device is
  **1.42 ms vs 57.43 ms CPU (~40x faster)**. Rerun in a separate warm-only
  harness the same day: 2.36 ms at n = 24, 1.36 ms at 200k, 5.24 ms at 1M.
* The stated *cause* was wrong too. The README blamed a per-call
  `DeviceContext()` construction costing "about 30 ms". There is a per-call
  `DeviceContext()`, but it does not cost 30 ms: the first call in a process
  costs 274 — 449 ms (CUDA context + module load) and later calls about 2 ms.
  The README's "1.47 ms isolated still did not beat 48.63 ms CPU once the 30 ms
  wrapper cost is added" is therefore not a real comparison.
* The default is still right for this package: a 200-eval TPE run makes **360
  `GMM1_lpdf` calls, all n = 24** (re-measured with a counting wrapper:
  `{24: 360}`), and at n = 24 the device is ~40x *slower*.

**No code was changed for this.** The device path is correct and opt-in; what
was wrong was the record of its cost. The README GPU section and
`bench/analysis.md` now both carry the measured numbers, and the
`GMM log-density GPU` benchmark row is labelled as a CPU fallback.

Device/CPU agreement at block boundaries (fresh process, gate stubbed):

| n | GMM max abs diff | log-GMM max abs diff |
| ---: | ---: | ---: |
| 1 | 3.93e-11 | 6.92e-10 |
| 255 | 7.10e-11 | 1.26e-09 |
| 256 | 7.14e-11 | 1.34e-09 |
| 257 | 7.15e-11 | 1.67e-09 |
| 4097 | 7.15e-11 | 1.82e-09 |
| 10003 | 7.15e-11 | 1.77e-09 |
| 200003 | 7.15e-11 | 1.85e-09 |

`tests/test_kernels.py` tolerates `rtol=1e-6, atol=3e-7` here, which covers
these; the README's "0.00e+00 at all of them" did not.

## 7. Threading claims

`mho_parallel_workers` re-read this session (requests 1, 2, 3, 4, 8, 32):

| n | k | pair gate | workers (request 4) |
| ---: | ---: | :---: | ---: |
| 12,000 | 48 | fail | 1 |
| 16,383 | 96 | pass | 1 |
| 16,384 | 24 / 25 / 48 / 96 | pass | 1 / 1 / 1 / 1 |
| 65,535 | 6 / 7 | fail / pass | 1 / 3 |
| 65,536 | 6 / 7 | fail / pass | 1 / 4 |
| 200,000 | 1 / 2 / 48 | fail / pass / pass | 1 / 4 / 4 |

So the pair clause (`n*k >= 393,216`) only binds between 16,384 and 65,535
samples; at `n = 16384` the `min(4, n // 16384)` clause pins it to 1 worker
even at `k = 96`, which contradicts the README's "(`n = 16384, k = 48` is
serial, `k = 96` is not)". Corrected in the README.

Serial vs threaded on the same call, best of 9, `tpe.PARALLEL_WORKERS` toggled
between runs, outputs `np.array_equal` every time:

| kernel, n (k = 48) | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| GMM 8,192 | 6.55 — 7.41 | 6.41 — 6.46 | 6.39 — 6.47 |
| GMM 12,000 | 9.43 — 9.45 | 9.52 — 9.59 | 9.53 — 9.54 |
| GMM 16,384 | 12.80 — 12.99 | 12.78 — 13.05 | 12.84 — 13.01 |
| GMM 32,768 | 25.86 — 25.89 | 19.63 — 19.78 | 19.44 — 20.27 |
| GMM 65,536 | 51.51 — 52.21 | 33.89 — 35.34 | 21.13 — 23.21 |
| GMM 200,000 | 157.47 — 158.40 | 87.49 — 88.65 | 49.46 — 49.54 |
| log-GMM 200,000 | 220.57 — 221.01 | 118.85 — 138.29 | 62.74 — 74.35 |

`anneal_bounds` (pool vs `_PARALLEL_WORKERS = 1`), best of 9, outputs equal:

| n | pool | serial |
| ---: | ---: | ---: |
| 1,000,000 | 2.40 ms | 3.98 ms |
| 5,000,000 | 10.67 ms | 41.03 ms |
| 20,000,000 | 42.90 ms | 202.91 ms |

## 8. Are the tests still asserting parity with upstream?

`git log -p -- tests/` over the whole history:

| commit | change to `tests/` |
| --- | --- |
| `c108a85` port | adds `conftest.py`, `test_api.py` (319 lines), `test_kernels.py` (204) |
| `a44a347` simd | adds `test_simd_lengths.py` (106), 0 deletions |
| `cf445c9` parallel | adds `test_parallel.py` (124), 0 deletions |
| `66063b7` gpu | adds 26 lines to `test_kernels.py`, 0 deletions |

There are **no deleted lines at all** in any commit that touches `tests/` —
`git log -p -- tests/` contains no `-` line matching `assert`, `rtol`, `atol`,
`allclose` or `array_equal`. Nothing was removed or loosened by an earlier
phase, so nothing was restored.

Coverage, as written:

* `test_adaptive_parzen_matches_upstream` and the SIMD-tail variant: against
  `hyperopt.tpe.adaptive_parzen_normal`, `rtol=1e-14, atol=1e-14`.
* `test_gmm_lpdf_matches_upstream` / `test_lgmm_lpdf_matches_upstream`, three
  bound/quantization modes each: against upstream at `rtol=1e-6, atol=3e-7`.
* `test_categorical_lpdf_matches_upstream`: `rtol=1e-8, atol=1e-8`.
* `test_linear_forgetting_matches_upstream`: `np.allclose` default tolerances.
* `test_simd_lengths.py`: every kernel at 0/1/W-1/W/W+1/4W/4W+1/97/100001 and
  component counts 1/W-1/W/W+1/3W/48 against upstream, same tolerances.
* `test_parallel.py`: `np.array_equal`, no tolerance, serial vs threaded.
* `test_public_signatures_match_upstream`: `inspect.signature` equality for
  `fmin`, `tpe.suggest`, `anneal.suggest`.

**One weakness, unchanged from the port commit and deliberately not
"fixed" here:** `test_search_quality_parity_with_upstream` computes both
losses and then asserts `our_loss < 0.2` and `upstream_loss < 0.2`; it never
compares the two to each other. Both optimizers are stochastic searches over
different RNG call sequences, so an equality assertion would be wrong, and the
density-level parity is covered by the kernel tests above. Measured this
session: TPE ours 0.0319 vs upstream 0.1512; anneal ours 0.00139 vs upstream
0.00245 — both branches of the package are doing real search work. The
assertion is weak in one direction only: a port that returned a *better* loss
than upstream would still pass. A port that returned `nan` or `inf` would fail,
because `nan < 0.2` and `inf < 0.2` are both `False`. It is a weak test, not a
broken one, and it is the original author's design choice. Flagged, not
changed.

## 9. What was changed

Only documentation. `git status` shows two modified files, no changes under
`src/`, `python/` or `tests/`:

* `README.md`
  * Performance table replaced with the medians of §2, with the protocol
    stated (three runs, load average, medians) and every figure in this session.
  * The SIMD "before" column removed and marked unreproducible; the "after",
    upstream and speedup columns replaced by measured values; the revert
    anecdotes marked as not re-measurable; the "code-truth" rows kept (each
    verified by reading the source) with their unmeasurable measurements
    labelled.
  * Parity bound corrected to 1.7e-8 (was "1.3e-8").
  * Parallel table: "before" column re-measured this session; the rows that
    missed 20% listed with claimed vs measured; a note on why the 4-thread
    figures are the least stable (shared host).
  * `anneal_bounds` pool-vs-serial table and its 1.7x/3.8x/4.7x reading.
  * Thread-scaling table re-measured (two rounds, ranges); the 32,768 row
    corrected (+45%); 8-thread claim corrected; worker-policy rule table added
    and the `k = 96` claim corrected.
  * GPU section rewritten: the paired table now reports cold and warm device
    calls; the "loses at every size" verdict replaced with "first call always
    loses, warm calls win from ~20k up"; the incorrect "30 ms per call" model
    removed; block-boundary agreement corrected to 7.2e-11 / 1.9e-9; the
    benchmark GPU row labelled as the CPU fallback it is; the one-GPU-call-per-
    process behaviour kept because it reproduced exactly (4309 → 3550 MiB).
  * Flops/byte gate annotated: the arithmetic values are kept, but the table's
    "candidate" verdict for `gmm_lpdf` is marked as not matching this session's
    measurements, with the measured CPU throughput (1.6 — 2.1 Gflop/s).
* `bench/analysis.md`: three in-place annotations marking the `17.33x` GPU rows
  as CPU-fallback measurements and pointing at this file. Its §1 CPU numbers
  were not restated as current because it is a Phase-0 historical document;
  the README carries the current figures.

### README numbers left deliberately unmeasured/unchanged

None. Every number in the rewritten Performance and GPU sections was measured
in this session. Two figures are labelled as *not re-measurable* rather than
replaced (the SIMD "before" column and the SIMD revert A/B timings), because no
build in this tree can produce them.

## 10. Final gate

```
pixi run build   -> exit 0
pixi run test    -> 354 passed, 6 skipped   (exit 0)
```

Same result as §1: the 6 skips are the GPU device cases skipped by the
4000 MiB gate after the first device context, each with the reason string.

## 11. Things a reader should not take on trust from this port

Stated plainly so they do not become folklore:

* The `GMM log-density GPU` table row is CPU work. It is labelled now, but
  anyone quoting "4.11x on GPU" is quoting a CPU number.
* The first `device="gpu"` call in a process costs 274 — 449 ms. Any caller
  that switches to the device expecting a win at n < 20,000 will lose badly.
* The 4-thread speedups (2.6 — 3.2x) are lower than the 3.4x the README used
  to quote, and they move with host load. Treat them as a range, not a point.
* `bench/analysis.md` is a Phase-0 document. Its §1 table is historical and its
  CPU figures are from the pre-unroll code (e.g. 499.6 ms for GMM where the
  current kernel is 58 ms threaded). Only the annotations added here are
  current.
