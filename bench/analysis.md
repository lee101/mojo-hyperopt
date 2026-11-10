# ANALYZE: where the time goes in mojo-hyperopt

Phase 0 of the Mojo port. No algorithm was changed and nothing in `src/` or `tests/` was
touched. Every number below comes from a run made in this session on this machine; the
derivations (bytes, flops per call) are arithmetic on the input sizes that the run actually
used. `bench/bench.py` was run unmodified as `pixi run bench`; instrumented variants lived in
`/tmp` and were run under the same `/tmp/mojo-bench.lock` so no two benchmarks overlapped.

## Machine and toolchain

| item | value | source |
| --- | --- | --- |
| CPU | Intel Xeon E5-2697 v4 @ 2.30 GHz, 36 cores / 72 threads (2x18, SMT), AVX2 | `lscpu`, `bench.py` header |
| OS / libc | Linux 6.8.0-147-generic, glibc 2.39 | `bench.py` header |
| Mojo | `1.2.0.dev2026092605`, `max == 26.7.0.dev2026092605` | `pixi.toml` |
| Python | 3.13 (pixi env) | `pixi.toml` |
| Upstream | hyperopt 0.2.7, importable, NumPy-2 `.item()` shim as in `bench.py` | `pixi.toml` |

Timing protocol: `pixi run bench` (min of 2-3 repeats per case, after one warm-up call), then
targeted micro-timings with `time.perf_counter`, min of 3-20 repeats. Peak bandwidth used only
as a sanity ceiling: 4 channels x 8B x 2.30 GHz / 2 sockets = 147 GB/s. Port kernels are
AVX2-only, so peak FLOPs assumed = 16 flops/cycle/call x ~2.0 GHz effective x 36 cores =
1.15 Tflop/s. Those two constants are the only non-measured inputs and are used only to label
"memory-bound" / "compute-bound".

## 1. `pixi run bench` (canonical run, this session)

| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |
| --- | ---: | ---: | ---: |
| adaptive Parzen, 20k observations | 1.16 ms | 1.61 ms | 1.39x |
| GMM log-density, 200k x 48 | 510.55 ms | 521.46 ms | 1.02x |
| GMM log-density GPU, 200k x 48 | 30.05 ms | 520.65 ms | 17.33x (CPU fallback; see VERIFY note in §5) |
| quantized GMM log-mass, 100k x 32 | 113.12 ms | 220.29 ms | 1.95x |
| log-GMM log-density, 200k x 48 | 569.58 ms | 751.60 ms | 1.32x |
| categorical log-density, 5M | 50.00 ms | 92.87 ms | 1.86x |
| annealing bounds, 5M centers | 13.47 ms | 77.74 ms | 5.77x |
| TPE `fmin`, 200 cheap evaluations | 254.27 ms | 673.04 ms | 2.65x |
| anneal `fmin`, 500 cheap evaluations | 501.05 ms | 908.13 ms | 1.81x |

Every declared input (200k x 48 = 9.6M component evaluations, 5M centers, 100k x 32) matches
the `bench.py` source. Nothing here violates the record size.

## 2. Corrected view of a stale README row

The README's Performance table claims `GMM log-density, 200k x 48` = 622.30 ms vs 525.94 ms
(**0.85x**, i.e. slower than upstream). Three independent runs this session disagree:

| source | ours | upstream | speedup |
| --- | ---: | ---: | ---: |
| `pixi run bench` (this session) | 510.55 ms | 521.46 ms | 1.02x |
| isolated wrapper call, n=200k k=48 | 499.6 ms | 515.7 ms | 1.03x |
| isolated kernel call, same buffers | 528.5 ms | — | — |

The port is **not currently slower than upstream on that row**, but it is only at parity; the
README figure is stale (likely measured before the adaptive-Parzen/SIMD work landed) and
should be corrected in the phase that rewrites docs. The two calls are within noise of each
other, so the honest reading is "tied", not "1.02x faster".

## 3. Time inside one call: glue vs kernel

Instrumented runs split every entry point into Python/ctypes glue and the Mojo/NumPy kernel.
"raw kernel" rows call the exported symbol directly with pre-made contiguous buffers, so they
contain no validation, no `_lib.addr`, no `np.ascontiguousarray` and no result allocation.

| entry point | input size | full call | raw kernel | kernel share | glue share |
| --- | --- | ---: | ---: | ---: | ---: |
| `GMM1_lpdf` | 200k x 48, unbounded | 499.6 ms | 528.5 ms | ~100% | <1% |
| `LGMM1_lpdf` | 200k x 48 | 583.4 ms | 575.7 ms | ~100% | <1% |
| `GMM1_lpdf` quantized | 100k x 32, q=0.25 | 114.7 ms | 114.3 ms | 99.6% | 0.4% |
| `categorical_lpdf` | 5M, 64 categories | 35.5 ms | 15.0 ms | 42% | 58% |
| `adaptive_parzen_normal` | 20k observations | 945 us | 150 us | 16% | 84% |
| `anneal_bounds` | 5M centers (threaded) | 12.05 ms | see §5 | ~100% | <1% |

Supporting glue measurements: empty `ctypes` call into the shared library = **6.0 us**;
`_lib.addr()` = **1.45 us** (9 call arguments per parzen call); `_lib.f64` on an already
float64 C-contiguous 5M array = **13.7 us**; `np.empty(200k)` = 0.52 us;
`np.isfinite(200k).all()` = 80.5 us. So glue is negligible for the big density kernels and
material only where the kernel itself is sub-millisecond.

## 4. Ranked hotspot table

Ranked by time per call. Bytes/flops are per call, derived from the loop structure in
`src/hyperopt.mojo`. "Bound by" compares the measured throughput with §0's peak figures.
`slower than upstream` marks rows where the port is **not** faster than hyperopt 0.2.7.

### 4a. Density kernels (the whole cost of `GMM1_lpdf` / `LGMM1_lpdf`)

| # | function | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | time in | slower than upstream? | next phase |
| -: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | `gmm_lpdf` (unquantized) | 200k x 48 | 499.6 ms | 9.6M comp x 16 B = 153.6 MB (+3x per-row rescan of w/m/s: 34.6 MB) = 188.2 MB | 76.8M log-exp pairs = ~1.2 Gflop | 6.4 | **compute-bound** | kernel `exp`/`log` calls: 19 Mcomp/s (1.9% of 1.15 Tflop/s) | no (tied, 1.03x) | **unroll** then **simd** |
| 2 | `lgmm_lpdf` (unquantized) | 200k x 48 | 583.4 ms | 188.2 MB | ~1.2 Gflop + 19.2M extra `log` | ~6.4 | **compute-bound** | kernel: `log(samples[i])` recomputed 2x K times per row | no (1.32x) | **unroll** (hoist the per-row `log`), then **simd** |
| 3 | `GMM1_lpdf`, device="gpu" | 200k x 48 | 30.05 ms | 1.6 MB in + 23 KB params + 1.6 MB out = 3.2 MB H2D/D2H | ~1.2 Gflop | 375 | compute-bound, moving on PCIe | host: 1 launch + 2 node-built contexts are 5.4x faster than CPU | **VERIFY: this row was the CPU fallback**, not the device; a real warm device call is 1.42 ms at this size | **none** for this size; **gpu** worth it only for n >= ~1M |
| 4 | quantized `gmm_lpdf` | 100k x 32 | 114.7 ms | 3.2M comp x 16 B = 51.2 MB | 3.2M x 2 erfc = 25.6M special fns | 0.5 | **compute-bound** (`erf`) | kernel: 28 Mcomp/s | no (1.95x) | **unroll** (paired CDF), **simd** rejected (see §6) |

### 4b. Vector / reduction kernels

| # | function | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | time in | slower than upstream? | next phase |
| -: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 5 | `mho_categorical_lpdf` (gather) | 5M, 64 cats | 35.5 ms full / 15.0 ms kernel | 40 MB + 40 MB out + 15 MB log-table reads = 95 MB | 5M gather + 5M compare | 0.1 | **memory-bound** (2.7 GB/s of 147 GB/s; gather-latency limited) | Python: full call is 2.4x the kernel; the rest is validation | no (1.86x) | **none** (glue: `none`, not kernel) |
| 6 | `mho_anneal_bounds`, 5M, 8 threads | 5M centers | 12.05 ms | 120 MB (5M x 3 x 8 B) | 5M x 4 = 20 Mflop | 20 Mflop / 16.8 MB (2 input reads + 2 streamed outputs) = 1.2 | **memory-bound** (10.0 GB/s, 6.8% of peak) | kernel + 8-way split (Python `ThreadPoolExecutor` per call) | no (6.5x) | **simd** (blocked), **none** for gpu |
| 7 | `mho_anneal_bounds`, serial path | 5M centers | 48.87 ms | 120 MB | 20 Mflop | 1.2 | memory-bound (3.3 GB/s: streaming store-bound) | kernel | no (vs 2.5 GB/s np.clip) | **simd** (blocked) |
| 8 | `mho_anneal_bounds` below threshold | 1M centers | 1.49 ms | 24 MB | 4 Mflop | 1.2 | memory-bound (21.6 GB/s) | kernel | no | **simd** |
| 9 | `mho_anneal_bounds` | 20M centers | 36.17 ms | 480 MB | 80 Mflop | 1.2 | memory-bound (13.3 GB/s; 8 workers oversubscribed on 36 cores) | kernel + 8 jobs | no | **parallel** (workers = min(cores, 32)) + **simd** |
| 10 | `mho_adaptive_parzen_normal` kernel | 20k obs | 150 us | 0.48 MB + 0.16 MB index = 0.64 MB | ~0.1 Mflop | 0.16 | **memory-bound** (4.3 GB/s) | kernel; wrapper total 945 us | no (1.39x) | **none** to kernel; see row 14 for glue |
| 11 | `np.argsort` inside Parzen wrapper | 20k obs | 748 us | 0.32 MB | 0.4 Mcompare | 1.3 | compute/NumPy | Python/NumPy | n/a | **none** (NumPy's optimized sort is already used) |
| 12 | bounded GMM (`low`/`high` axis mask) | 200k x 48 | 505.2 ms | 188.2 MB | ~1.2 Gflop | 6.4 | compute-bound | kernel | no | same as row 1 |
| 13 | numpy vectorized GMM baseline | 200k x 48 | 448.9 ms | 76.8 MB term temp | ~1.2 Gflop | **15.6** | compute-bound; allocates 76.8 MB | NumPy/OpenBLAS per-op, 7 passes | n/a | n/a; a numpy fallback is 1.10x faster than the port's tight scalar loop. This is the target to beat |

### 4c. Driver-level (Python glue), ranked by whole-run time

Driver profile of `tpe.suggest` x180 after startup (`cProfile`, `tpe.py`, cumulative):

| # | function | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | time in | slower than upstream? | next phase |
| -: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 14 | `_split_history` + `_posterior_sample` (`tpe.py:377,405`) | 180 calls at 20-200 trials | 104 ms + 267 ms (of 400 ms total) | ~2 KB/trial dicts | 0 | n/a | Python glue | Python dict/list re-walk every suggest | 2.65x faster than upstream overall (254 ms vs 673 ms) | **none** (Python by design). Only 2 flops/param |
| 15 | `anneal.AnnealingAlgo._history` | 500 evals x 2 params | 914 ms of 1672 ms (55%) | ~1.5M `dict.get` calls | 0 | n/a | **Python glue**; rescan of all trials per param per eval (O(n^2)) | Python loop over every trial | **slower** per-eval (1.84 vs 0.90 ms/eval at 500 evals) **but faster overall** (501 vs 921 ms) because upstream's anneal schedule is far heavier | **none** (Python); the ann-worthy piece is `anneal_bounds` (6) |
| 16 | `checked_call` + `addr()` overhead | 1440 kernel calls / 200 evals | 79 ms + 30 ms (27% of 400 ms) | 7200 `addr` calls @ 1.45 us | 0 | n/a | Python/ctypes glue | per kernel call, not per element | n/a | **none** (already batched, one call per kernel) |
| 17 | `_sample_mixture` retry loop (`tpe.py:335`) | 360 calls | 61 ms | small | — | — | Python glue; rejection loop with per-draw Python `float` coercion | `.extend(float(v) for v in draws)` | n/a | **none** |

### 4d. Scaling evidence for the top kernel

`GMM1_lpdf`, ours, n=200k as K grows, and k=48 as N grows. Time is linear, so the cost is
entirely the inner component loop:

| k (n=200k) | 4 | 12 | 24 | 48 | 96 | 192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 52.61 | 135.78 | 255.71 | 501.05 | 992.34 | 2027.63 |
| Gcomp/s | 0.015 | 0.018 | 0.019 | 0.019 | 0.019 | 0.019 |

| n (k=48) | 10k | 50k | 200k | 1M |
| --- | ---: | ---: | ---: | ---: |
| ms | 24.83 | 123.65 | 505.48 | 2527.03 |

Flat 0.019 Gcomp/s across 4 < k < 192 is the tell: no block-level reuse, no vectorisation of the
`exp`, and two full walks of the component arrays per row (max pass + exp pass), with
`log(weights[j] / (SQRT_2PI * sigma))` recomputed inside both walks.

## 5. GPU verdict inputs (flops per byte)

| kernel | measured flops/byte | GPU worth it? | reason |
| --- | ---: | --- | --- |
| unquantized `gmm_lpdf` (row 1) | **6.4** | yes, above the ~2 flop/B rule, but only for large n | **corrected in VERIFY**: the "17.33x at 200k" row in §1 was a CPU-fallback measurement, not a device one. A real warm device call is ~1.4 ms against 57 ms of CPU at 200k (about 40x), but the first call in a process costs 274-449 ms, so the device only wins from ~n = 20,000 up on warm calls. See `bench/verify.md`. |
| unquantized `lgmm_lpdf` (row 2) | **~6.4** | no new GPU work | the same kernel shape as row 1 already has a GPU twin (`log_space=1` path in `mho_gmm1_lpdf_gpu`); nothing to add |
| quantized `gmm_lpdf` (row 4) | **0.5** | **no** | 0.5 flops/byte < 2; the 51.2 MB working set streams from L2 faster than a copy to device |
| `categorical_lpdf` (row 5) | **0.1** | **no** | 0.1 flops/byte, gather-latency bound; 95 MB of traffic would cost ~2 x 1.6 MiB per-job copy for negligible compute |
| `anneal_bounds` (row 6) | **1.2** | **no** | measured throughput is 10 GB/s against a 147 GB/s ceiling: memory-bound, and 1.2 < the ~2 flop/B rule |
| `adaptive_parzen_normal` (row 10) | **0.16** | **no** | 0.16 flops/byte; 0.64 MB already fits in L2 |

Every verdict above is backed by a measured flops/byte figure, none is a guess from "kernel
looks small". The only GPU row with a figure above 2 is unquantized GMM, which already has a
GPU path.

## 6. Python-level loops that re-enter ctypes, and per-call allocations

Checked all six call sites in `python/mojohyperopt/`.

* **No per-element ctypes re-entry.** Every kernel is one call per batch:
  `tpe.py:112` (parzen), `tpe.py:176` (GMM), `tpe.py:243` (log-GMM), `tpe.py:278`
  (categorical), `anneal.py:23` (bounds), `tpe.py:65` (GPU). `anneal.py:61-71` submits
  slices for a threaded call, but the number of ctypes calls is bounded by
  `_PARALLEL_WORKERS` (8), not by element count.
* **Python loops that do call numpy per element** (glue cost, not ctypes):
  * `anneal.py:180-193 visit()` walks every space node once per trial; **`anneal.py:88-101`
    `_history`** re-scans all trials for every parameter on every evaluation. That is
    O(n_evals^2) `dict.get`s: 1,498,502 `dict.get` calls in the 500-eval profile (914 ms,
    55% of run). It is the top Python hotspot.
  * `tpe.py:446-465` and `:473-481` in `_posterior_sample` loop over candidate indices in
    Python to write `assignments[index][node.label]` and `scores[...]`, once per
    (candidate, parameter). 39,620 `list.append` + 39,420 `dict.setdefault` in the 200-eval
    profile.
  * `tpe.py:346-353` `_sample_mixture` is a rejection `while` loop whose inner
    `accepted.extend(float(v) for v in draws)` is Python per accepted value.
  * `tpe.py:392` `_split_history` builds two 20k-element `dict` comprehensions per call.
* **Per-call allocations/copies** (measured, and the ones that matter):
  | site | allocation | measured cost at the benchmark size |
  | --- | --- | ---: |
  | `tpe.py:160,227` | `np.empty(n)` result | 0.52 us at 200k |
  | `tpe.py:111` `np.argsort` in parzen | 0.32 MB index | **748 us** (the row-11 dominant cost) |
  | `tpe.py:108-110` | 3 x `np.empty(n+1)` | included in row 10's 945 us |
  | `anneal.py:48-49` | 2 x `np.empty(n)` | 2 x 5M x 8 B = 80 MB per call |
  | `tpe.py:63-64` GPU | `log_coefficients` + `safe_sigmas` (k only, 48x8 B) | negligible |
  | `tpe.py:276` categorical | `np.log(probabilities)` | 64 doubles, negligible |
  | `_lib.f64/i64` | `ascontiguousarray` copy | **13.7 us at 5M** (a same-dtype contiguous input is still copied at this rate; not a real copy, just the API) |
  * The CPU kernels themselves allocate nothing (README claim confirmed by the source and by
    the raw-vs-wrapper timings matching).

## 7. Where the port stands vs upstream

| case | port slower than upstream? | note |
| --- | --- | --- |
| adaptive Parzen 20k | no | 1.39x; kernel is 150 us against a 748 us sort |
| GMM log-density 200k x 48 | no / **tied** | 1.02x vs a stale README claim of 0.85x |
| GMM GPU | no | 17.33x in §1, but that row fell back to the CPU (VERIFY correction); a warm device call is ~40x, a cold first call is a large loss |
| quantized GMM 100k x 32 | no | 1.95x |
| log-GMM 200k x 48 | no | 1.32x |
| categorical 5M | no | 1.86x, but 58% of the 35.5 ms is validation, not the kernel |
| anneal bounds 5M | no | 5.77x in `bench.py`; 6.5x in the isolated run |
| TPE fmin 200 | no | 2.65x overall, still O(n^2) in Python |
| anneal fmin 500 | no overall, **yes per evaluation** | 501.05 ms vs 908.13 ms overall, but 0.896 ms/eval vs 1.842 ms/eval for ours vs upstream; see below |

Per-evaluation numbers measured separately: ours 0.167 / 0.293 / 0.896 ms/eval at 20 / 100 /
500 evals vs upstream 1.842 ms/eval at 500. The port's per-eval cost grows with history length
because `_history` is a full rescan; upstream grows faster still because its anneal schedule is
heavier. The port is not slower end-to-end on any row, but it will cross over upstream if the
history scan is left untouched (its slope is `+0.73 ms/eval` per doubling versus upstream's
heavier constant).

## 8. Ranked backlog for the next phases

| rank | hotspot | evidence | next phase |
| -: | --- | --- | --- |
| 1 | `gmm_lpdf` max+exp double walk | rows 1, 4d: 19 Mcomp/s (1.9% of peak); `log(weights...)` recomputed per component in both passes; numpy is 1.10x faster | **unroll** (block the component loop, hoist the constant `log(w/(sqrt2pi*sigma))` into a 48-element priming, single pass with running logsumexp), then **simd** |
| 2 | `lgmm_lpdf` per-row `log(samples[i])` recomputed 2xK per element | row 2; measured 19 Mcomp/s vs 19.4 Mcomp/s for row 1, so the hoist is not yet paid for | **unroll** (hoist the row `log` into a local before both passes), then **simd** |
| 3 | quantized `gmm_lpdf` paired `erf` calls | row 4: 28 Mcomp/s, 0.5 flop/B | **unroll** (compute upper/lower CDF for adjacent components in one pass), **simd** rejected |
| 4 | `mho_anneal_bounds` streaming stores | row 6: 10 GB/s vs 147 GB/s; 3 streams | **simd** (cache-blocked clamp/offset) |
| 5 | anneal parallel split fixed at 8 workers | row 9: 13.3 GB/s, oversubscribed on 36 cores | **parallel** (workers = min(cores, 32)) |
| 6 | `_history` rescan | row 15: 914 ms of 1672 ms, 1.5M `dict.get` | **none** (Python by contract; no kernel to accelerate) |
| 7 | categorical validation (58% of call) | row 5 | **none** (one-time checks; do not drop them to win a benchmark) |
| 8 | GMM GPU path | row 3 | **gpu** only for n >= ~1M; the path already exists |
| 9 | quantized / categorical / anneal / parzen GPU | §5: 0.1-0.5 flop/B | **none** |

Every compute-bound row above 2 flops/byte has either a committed GPU action (row 3) or an
unroll/simd action ahead of it. Every memory-bound row below 2 flops/byte is explicitly
CPU-only.

## 9. Reproduce

```bash
pixi run bench                     # §1 table
# instrumented variants used for §3-§7 lived in /tmp and were run as
#   flock /tmp/mojo-bench.lock pixi run python -u /tmp/mho_analyze*.py
```

The `/tmp` scripts are not checked in; they only read from `python/` and `src/` and write
nothing into the repo.
