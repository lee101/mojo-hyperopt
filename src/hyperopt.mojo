"""Numerical kernels for TPE density estimation and annealing proposals."""

# Parallelism lives in `max` on this toolchain; the `std` forms do not compile.
from max.algorithm import parallelize
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import erf, exp, log, sqrt
from std.sys import simd_width_of


comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime EPS = 1.0e-12
comptime SQRT_2 = 1.4142135623730950488
comptime SQRT_2PI = 2.5066282746310005024
# The quantized line integral walks LANES components at a time with independent
# partial sums, combined once at the end: a single running sum serialises on
# floating-point add latency. The unquantized rows use W-lane SIMD accumulators.
comptime LANES = 8
comptime W = simd_width_of[DType.float64]()
comptime MODE_GMM = 0
comptime MODE_LOG = 1
# Samples per thread: the density kernels are compute-bound at ~19 Mcomp/s on a
# single core, so the smallest run worth starting threads for is the one where
# each worker still gets enough rows to amortise its launch. Measured crossover
# at k=48 is between 12k and 16k samples; the gate below is 16k with margin.
comptime PARALLEL_MIN_SAMPLES = 16384
# The host is shared. `choose_workers` therefore never hands out more than 4
# workers, however many are requested; the larger constant is only a bound on
# the request so the 1/2/4-thread split can be forced explicitly. A serial call
# (1), the 2- and 4-thread prefixes of a larger pool and the capped value all
# produce identical bytes, so nothing below depends on which value is picked.
comptime PARALLEL_WORKERS_MAX = 32
comptime PARALLEL_WORKERS_CAP = 4
comptime PARALLEL_MIN_WORK = 393216
# Each worker is handed whole blocks of BLOCK_ROWS_MULT x W rows. Block sizes
# are multiples of W, so every worker's first SIMD load keeps 16-byte alignment,
# and the block boundaries -- not the worker count -- decide which rows go where.
# The block size is therefore part of the output contract: threaded and serial
# calls produce bitwise-identical results for any worker count.
comptime BLOCK_ROWS_MULT = 16

def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def normal_cdf(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    var denom = SQRT_2 * sigma
    if denom < EPS:
        denom = EPS
    return 0.5 * (1.0 + erf((x - mu) / denom))


def normal_log_term(x: Float64, mu: Float64, weight: Float64, sigma: Float64) -> Float64:
    var z = (x - mu) / sigma
    return -0.5 * z * z + log(weight / (SQRT_2PI * sigma))


def lognormal_log_term(x: Float64, log_x: Float64, mu: Float64, weight: Float64, sigma: Float64) -> Float64:
    var z = (log_x - mu) / sigma
    return -0.5 * z * z + log(weight) - log(SQRT_2PI * sigma * x)


def component_log_terms[mode: Int](
    x: Float64,
    log_x: Float64,
    mus: FPtr,
    weights: FPtr,
    sigmas: FPtr,
    j: Int,
) -> SIMD[DType.float64, W]:
    """W component log-terms starting at j, in the selected density space.

    The log-normal form reproduces `lognormal_log_term` operation for operation;
    the per-row `log(x)` is computed once and passed in.
    """
    var mu = mus.unsafe_load[width=W](j)
    var sg = sigmas.unsafe_load[width=W](j)
    var w = weights.unsafe_load[width=W](j)
    var xv = SIMD[DType.float64, W](x)
    var lxv = SIMD[DType.float64, W](log_x)
    if mode == MODE_LOG:
        var z = (lxv - mu) / sg
        return -0.5 * z * z + log(w) - log(SQRT_2PI * sg * xv)
    var z2 = (xv - mu) / sg
    return -0.5 * z2 * z2 + log(w / (SQRT_2PI * sg))


def logsumexp_row[mode: Int](
    mus: FPtr,
    weights: FPtr,
    sigmas: FPtr,
    k: Int,
    x: Float64,
) -> Float64:
    """Return log(sum_j exp(term_j)) for one sample, SIMD over components.

    The largest term is found first, then the exponentials are summed relative
    to it, so nothing overflows and the terms are formed twice but exponentiated
    once. Folding the two passes into one was measured slower at caller sizes:
    a single fused pass has to exponentiate every component twice (once to fold
    it into the running maximum), while the second maximum pass here only ever
    builds cheap underflowing exponentials. Four accumulators keep the sum from
    serialising on floating-point add latency.

    Buffers are the contiguous NumPy arrays handed through the FFI, so the
    W-lane loads start on the allocation's 16-byte boundary.
    """
    # log(x) is hoisted: the two-pass form below would otherwise recompute it
    # once per component per pass.
    var row_log_x = x
    if mode == MODE_LOG:
        row_log_x = log(x)
    var floor = SIMD[DType.float64, W](-1.0e300)
    var j = 0
    var best = floor
    while j + W <= k:
        best = max(best, component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j))
        j += W
    var max_term = best.reduce_max()
    while j < k:
        var term = normal_log_term(x, mus[unsafe_offset=j], weights[unsafe_offset=j], sigmas[unsafe_offset=j])
        if mode == MODE_LOG:
            term = lognormal_log_term(x, row_log_x, mus[unsafe_offset=j], weights[unsafe_offset=j], sigmas[unsafe_offset=j])
        max_term = max(max_term, term)
        j += 1
    var top = SIMD[DType.float64, W](max_term)
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = acc0
    var acc2 = acc0
    var acc3 = acc0
    j = 0
    while j + 4 * W <= k:
        acc0 += exp(component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j) - top)
        acc1 += exp(component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j + W) - top)
        acc2 += exp(component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j + 2 * W) - top)
        acc3 += exp(component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j + 3 * W) - top)
        j += 4 * W
    while j + W <= k:
        acc0 += exp(component_log_terms[mode](x, row_log_x, mus, weights, sigmas, j) - top)
        j += W
    var total = ((acc0 + acc1) + (acc2 + acc3)).reduce_add()
    while j < k:
        var term = normal_log_term(x, mus[unsafe_offset=j], weights[unsafe_offset=j], sigmas[unsafe_offset=j])
        if mode == MODE_LOG:
            term = lognormal_log_term(x, row_log_x, mus[unsafe_offset=j], weights[unsafe_offset=j], sigmas[unsafe_offset=j])
        total += exp(term - max_term)
        j += 1
    return log(total) + max_term


def clamp_sigmas(sigmas: FPtr, k: Int):
    """Replace out-of-range sigmas once per call instead of once per element."""
    for j in range(k):
        if sigmas[unsafe_offset=j] < EPS:
            sigmas[unsafe_offset=j] = EPS


def clamp_sigmas_addr(sigmas_addr: Int, k: Int):
    clamp_sigmas(FPtr(unsafe_from_address=sigmas_addr), k)


def choose_workers(n: Int, k: Int, requested: Int) -> Int:
    """Worker count for one call, independent of how many threads the runtime has.

    A contract rather than a tuning knob: every build and every machine picks
    the same split, which is what lets the serial and threaded results be
    compared bitwise and keeps the test off the host's core count. Below
    4 * PARALLEL_MIN_SAMPLES the request is only honoured at 1/2/4 (never more
    than the call has blocks for); above it the 4-worker cap applies.
    """
    if n * k < PARALLEL_MIN_WORK:
        return 1
    var threads = min(requested, PARALLEL_WORKERS_MAX)
    if n < PARALLEL_WORKERS_CAP * PARALLEL_MIN_SAMPLES:
        # Only the 1/2/4 prefix of any larger request is honoured here.
        threads = min(threads, PARALLEL_WORKERS_CAP)
        threads = min(threads, n // PARALLEL_MIN_SAMPLES)
    return max(1, min(threads, PARALLEL_WORKERS_CAP))


def acceptance(
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    low: Float64,
    high: Float64,
    has_low: Bool,
    has_high: Bool,
) -> Float64:
    if not has_low and not has_high:
        return 1.0
    var total = Float64(0.0)
    for j in range(k):
        var lower = Float64(0.0)
        var upper = Float64(1.0)
        if has_low:
            lower = normal_cdf(low, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
        if has_high:
            upper = normal_cdf(high, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
        total += weights[unsafe_offset=j] * (upper - lower)
    return total


def gmm_lpdf_rows(
    samples: FPtr,
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    log_p_accept: Float64,
    result: FPtr,
    first: Int,
    last: Int,
):
    """Rows `first:last` of the unquantized GMM log-density.

    A whole-row function so the serial call and every parallel worker run the
    identical per-row arithmetic; splitting here cannot change a single bit.
    """
    for i in range(first, last):
        result[unsafe_offset=i] = logsumexp_row[MODE_GMM](
            mus, weights, sigmas, k, samples[unsafe_offset=i]
        ) - log_p_accept


def gmm_lpdf(
    samples: FPtr,
    n: Int,
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    low: Float64,
    high: Float64,
    has_low: Bool,
    has_high: Bool,
    q: Float64,
    quantized: Bool,
    result: FPtr,
    workers: Int,
):
    var p_accept = acceptance(
        weights, mus, sigmas, k, low, high, has_low, has_high
    )
    var log_p_accept = log(p_accept)
    if not quantized:
        clamp_sigmas(sigmas, k)
        # Blocks are whole multiples of W rows, so every worker's first SIMD
        # load keeps the 16-byte alignment the W-lane loads rely on.
        var block_rows = BLOCK_ROWS_MULT * W
        var blocks = (n + block_rows - 1) // block_rows
        var kern = k
        var rows = n
        var lpa = log_p_accept
        var step = block_rows
        var count = blocks
        if workers <= 1 or blocks <= 1:
            gmm_lpdf_rows(samples, weights, mus, sigmas, k, log_p_accept, result, 0, n)
            return
        def run_block(block: Int) {imm}:
            var start = block * step
            gmm_lpdf_rows(
                samples, weights, mus, sigmas, kern, lpa, result,
                start, min(start + step, rows),
            )
        parallelize(run_block, count, workers)
        return

    for i in range(n):
        var lower = samples[unsafe_offset=i] - 0.5 * q
        var upper = samples[unsafe_offset=i] + 0.5 * q
        if has_low and lower < low:
            lower = low
        if has_high and upper > high:
            upper = high
        var a0 = Float64(0.0)
        var a1 = Float64(0.0)
        var a2 = Float64(0.0)
        var a3 = Float64(0.0)
        var a4 = Float64(0.0)
        var a5 = Float64(0.0)
        var a6 = Float64(0.0)
        var a7 = Float64(0.0)
        var j = 0
        while j + LANES <= k:
            a0 += weights[unsafe_offset=j] * (normal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j]) - normal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j]))
            a1 += weights[unsafe_offset=j + 1] * (normal_cdf(upper, mus[unsafe_offset=j + 1], sigmas[unsafe_offset=j + 1]) - normal_cdf(lower, mus[unsafe_offset=j + 1], sigmas[unsafe_offset=j + 1]))
            a2 += weights[unsafe_offset=j + 2] * (normal_cdf(upper, mus[unsafe_offset=j + 2], sigmas[unsafe_offset=j + 2]) - normal_cdf(lower, mus[unsafe_offset=j + 2], sigmas[unsafe_offset=j + 2]))
            a3 += weights[unsafe_offset=j + 3] * (normal_cdf(upper, mus[unsafe_offset=j + 3], sigmas[unsafe_offset=j + 3]) - normal_cdf(lower, mus[unsafe_offset=j + 3], sigmas[unsafe_offset=j + 3]))
            a4 += weights[unsafe_offset=j + 4] * (normal_cdf(upper, mus[unsafe_offset=j + 4], sigmas[unsafe_offset=j + 4]) - normal_cdf(lower, mus[unsafe_offset=j + 4], sigmas[unsafe_offset=j + 4]))
            a5 += weights[unsafe_offset=j + 5] * (normal_cdf(upper, mus[unsafe_offset=j + 5], sigmas[unsafe_offset=j + 5]) - normal_cdf(lower, mus[unsafe_offset=j + 5], sigmas[unsafe_offset=j + 5]))
            a6 += weights[unsafe_offset=j + 6] * (normal_cdf(upper, mus[unsafe_offset=j + 6], sigmas[unsafe_offset=j + 6]) - normal_cdf(lower, mus[unsafe_offset=j + 6], sigmas[unsafe_offset=j + 6]))
            a7 += weights[unsafe_offset=j + 7] * (normal_cdf(upper, mus[unsafe_offset=j + 7], sigmas[unsafe_offset=j + 7]) - normal_cdf(lower, mus[unsafe_offset=j + 7], sigmas[unsafe_offset=j + 7]))
            j += LANES
        while j < k:
            a0 += weights[unsafe_offset=j] * (normal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j]) - normal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j]))
            j += 1
        result[unsafe_offset=i] = log((((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)))) - log_p_accept


def gmm_lpdf_gpu_kernel(
    samples: FPtr,
    n: Int32,
    log_coefficients: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int32,
    log_space: Int32,
    result: FPtr,
):
    # Kernel arguments must be DevicePassable: 1.2.0 rejects index-sized Int.
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var sample = samples[unsafe_offset=i]
    var transformed = log(sample) if log_space != 0 else sample
    var jacobian = transformed if log_space != 0 else 0.0
    var max_term = -1.0e300
    for j in range(k):
        var sigma = sigmas[unsafe_offset=j]
        if sigma < EPS:
            sigma = EPS
        var z = (transformed - mus[unsafe_offset=j]) / sigma
        var term = -0.5 * z * z + log_coefficients[unsafe_offset=j] - jacobian
        if term > max_term:
            max_term = term
    var total = Float64(0.0)
    for j in range(k):
        var sigma = sigmas[unsafe_offset=j]
        if sigma < EPS:
            sigma = EPS
        var z = (transformed - mus[unsafe_offset=j]) / sigma
        var term = -0.5 * z * z + log_coefficients[unsafe_offset=j] - jacobian
        total += exp(term - max_term)
    result[unsafe_offset=i] = log(total) + max_term


def gmm_lpdf_gpu(
    samples: FPtr,
    n: Int,
    log_coefficients: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    log_space: Int,
    result: FPtr,
) raises:
    var ctx = DeviceContext()
    var device_samples = ctx.enqueue_create_buffer[DType.float64](n)
    var device_coefficients = ctx.enqueue_create_buffer[DType.float64](k)
    var device_mus = ctx.enqueue_create_buffer[DType.float64](k)
    var device_sigmas = ctx.enqueue_create_buffer[DType.float64](k)
    var device_result = ctx.enqueue_create_buffer[DType.float64](n)
    ctx.enqueue_copy(device_samples, samples)
    ctx.enqueue_copy(device_coefficients, log_coefficients)
    ctx.enqueue_copy(device_mus, mus)
    ctx.enqueue_copy(device_sigmas, sigmas)
    comptime BLOCK_SIZE = 256
    var blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    ctx.enqueue_function[gmm_lpdf_gpu_kernel](
        device_samples,
        Int32(n),
        device_coefficients,
        device_mus,
        device_sigmas,
        Int32(k),
        Int32(log_space),
        device_result,
        grid_dim=blocks,
        block_dim=BLOCK_SIZE,
    )
    ctx.enqueue_copy(result, device_result)
    ctx.synchronize()


def lognormal_cdf(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var denom = SQRT_2 * sigma
    if denom < EPS:
        denom = EPS
    return 0.5 + 0.5 * erf((log(x if x > EPS else EPS) - mu) / denom)


def lgmm_lpdf_rows(
    samples: FPtr,
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    result: FPtr,
    first: Int,
    last: Int,
):
    """Rows `first:last` of the unquantized log-GMM log-density (no jacobian)."""
    for i in range(first, last):
        result[unsafe_offset=i] = logsumexp_row[MODE_LOG](
            mus, weights, sigmas, k, samples[unsafe_offset=i]
        )


def lgmm_lpdf(
    samples: FPtr,
    n: Int,
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    k: Int,
    low: Float64,
    high: Float64,
    has_low: Bool,
    has_high: Bool,
    q: Float64,
    quantized: Bool,
    result: FPtr,
    workers: Int,
):
    var p_accept = acceptance(
        weights, mus, sigmas, k, low, high, has_low, has_high
    )
    var log_p_accept = log(p_accept)
    if not quantized:
        clamp_sigmas(sigmas, k)
        var block_rows = BLOCK_ROWS_MULT * W
        var blocks = (n + block_rows - 1) // block_rows
        var kern = k
        var rows = n
        var step = block_rows
        var count = blocks
        if workers <= 1 or blocks <= 1:
            lgmm_lpdf_rows(samples, weights, mus, sigmas, k, result, 0, n)
            return
        def run_block(block: Int) {imm}:
            var start = block * step
            lgmm_lpdf_rows(
                samples, weights, mus, sigmas, kern, result,
                start, min(start + step, rows),
            )
        parallelize(run_block, count, workers)
        return

    for i in range(n):
        var lower = samples[unsafe_offset=i] - 0.5 * q
        var upper = samples[unsafe_offset=i] + 0.5 * q
        if lower < 0.0:
            lower = 0.0
        if has_low:
            var floor = exp(low)
            if lower < floor:
                lower = floor
        if has_high:
            var ceiling = exp(high)
            if upper > ceiling:
                upper = ceiling
        var a0 = Float64(0.0)
        var a1 = Float64(0.0)
        var a2 = Float64(0.0)
        var a3 = Float64(0.0)
        var a4 = Float64(0.0)
        var a5 = Float64(0.0)
        var a6 = Float64(0.0)
        var a7 = Float64(0.0)
        var j = 0
        while j + LANES <= k:
            a0 += weights[unsafe_offset=j] * (lognormal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j]) - lognormal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j]))
            a1 += weights[unsafe_offset=j + 1] * (lognormal_cdf(upper, mus[unsafe_offset=j + 1], sigmas[unsafe_offset=j + 1]) - lognormal_cdf(lower, mus[unsafe_offset=j + 1], sigmas[unsafe_offset=j + 1]))
            a2 += weights[unsafe_offset=j + 2] * (lognormal_cdf(upper, mus[unsafe_offset=j + 2], sigmas[unsafe_offset=j + 2]) - lognormal_cdf(lower, mus[unsafe_offset=j + 2], sigmas[unsafe_offset=j + 2]))
            a3 += weights[unsafe_offset=j + 3] * (lognormal_cdf(upper, mus[unsafe_offset=j + 3], sigmas[unsafe_offset=j + 3]) - lognormal_cdf(lower, mus[unsafe_offset=j + 3], sigmas[unsafe_offset=j + 3]))
            a4 += weights[unsafe_offset=j + 4] * (lognormal_cdf(upper, mus[unsafe_offset=j + 4], sigmas[unsafe_offset=j + 4]) - lognormal_cdf(lower, mus[unsafe_offset=j + 4], sigmas[unsafe_offset=j + 4]))
            a5 += weights[unsafe_offset=j + 5] * (lognormal_cdf(upper, mus[unsafe_offset=j + 5], sigmas[unsafe_offset=j + 5]) - lognormal_cdf(lower, mus[unsafe_offset=j + 5], sigmas[unsafe_offset=j + 5]))
            a6 += weights[unsafe_offset=j + 6] * (lognormal_cdf(upper, mus[unsafe_offset=j + 6], sigmas[unsafe_offset=j + 6]) - lognormal_cdf(lower, mus[unsafe_offset=j + 6], sigmas[unsafe_offset=j + 6]))
            a7 += weights[unsafe_offset=j + 7] * (lognormal_cdf(upper, mus[unsafe_offset=j + 7], sigmas[unsafe_offset=j + 7]) - lognormal_cdf(lower, mus[unsafe_offset=j + 7], sigmas[unsafe_offset=j + 7]))
            j += LANES
        while j < k:
            a0 += weights[unsafe_offset=j] * (lognormal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j]) - lognormal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j]))
            j += 1
        result[unsafe_offset=i] = log((((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)))) - log_p_accept


def forgetting_weight(index: Int, n: Int, lf: Int) -> Float64:
    if lf <= 0 or n < lf:
        return 1.0
    var ramp = n - lf
    if index >= ramp:
        return 1.0
    if ramp == 1:
        return 1.0 / Float64(n)
    return 1.0 / Float64(n) + (1.0 - 1.0 / Float64(n)) * Float64(
        index
    ) / Float64(ramp - 1)


def adaptive_parzen(
    observations: FPtr,
    n: Int,
    prior_weight: Float64,
    prior_mu: Float64,
    prior_sigma: Float64,
    lf: Int,
    weights: FPtr,
    mus: FPtr,
    sigmas: FPtr,
    order: IPtr,
):
    if n == 0:
        weights[unsafe_offset=0] = 1.0
        mus[unsafe_offset=0] = prior_mu
        sigmas[unsafe_offset=0] = prior_sigma
        return

    var prior_pos = 0
    while prior_pos < n and observations[unsafe_offset=Int(order[unsafe_offset=prior_pos])] < prior_mu:
        prior_pos += 1

    for j in range(prior_pos):
        mus[unsafe_offset=j] = observations[unsafe_offset=Int(order[unsafe_offset=j])]
    mus[unsafe_offset=prior_pos] = prior_mu
    for j in range(prior_pos, n):
        mus[unsafe_offset=j + 1] = observations[unsafe_offset=Int(order[unsafe_offset=j])]

    var count = n + 1
    if count == 2:
        sigmas[unsafe_offset=0] = prior_sigma
        sigmas[unsafe_offset=1] = prior_sigma
        if prior_pos == 0:
            sigmas[unsafe_offset=1] = prior_sigma * 0.5
        else:
            sigmas[unsafe_offset=0] = prior_sigma * 0.5
    else:
        sigmas[unsafe_offset=0] = mus[unsafe_offset=1] - mus[unsafe_offset=0]
        sigmas[unsafe_offset=count - 1] = mus[unsafe_offset=count - 1] - mus[unsafe_offset=count - 2]
        comptime W = simd_width_of[DType.float64]()
        var j = 1
        var vector_end = 1 + ((count - 2) // W) * W
        while j < vector_end:
            var center = mus.unsafe_load[width=W](j)
            var left = center - mus.unsafe_load[width=W](j - 1)
            var right = mus.unsafe_load[width=W](j + 1) - center
            sigmas.unsafe_store(j, max(left, right))
            j += W
        while j < count - 1:
            sigmas[unsafe_offset=j] = max(mus[unsafe_offset=j] - mus[unsafe_offset=j - 1], mus[unsafe_offset=j + 1] - mus[unsafe_offset=j])
            j += 1

    for j in range(prior_pos):
        weights[unsafe_offset=j] = forgetting_weight(Int(order[unsafe_offset=j]), n, lf)
    weights[unsafe_offset=prior_pos] = prior_weight
    for j in range(prior_pos, n):
        weights[unsafe_offset=j + 1] = forgetting_weight(Int(order[unsafe_offset=j]), n, lf)

    var min_divisor = min(100.0, Float64(1 + count))
    var min_sigma = prior_sigma / min_divisor
    comptime W = simd_width_of[DType.float64]()
    var j = 0
    var vector_end = (count // W) * W
    var min_sigma_vec = SIMD[DType.float64, W](min_sigma)
    var prior_sigma_vec = SIMD[DType.float64, W](prior_sigma)
    while j < vector_end:
        var sigma = sigmas.unsafe_load[width=W](j)
        sigmas.unsafe_store(j, max(min_sigma_vec, min(prior_sigma_vec, sigma)))
        j += W
    while j < count:
        sigmas[unsafe_offset=j] = max(min_sigma, min(prior_sigma, sigmas[unsafe_offset=j]))
        j += 1
    sigmas[unsafe_offset=prior_pos] = prior_sigma

    var total = Float64(0.0)
    j = 0
    while j < vector_end:
        total += weights.unsafe_load[width=W](j).reduce_add()
        j += W
    while j < count:
        total += weights[unsafe_offset=j]
        j += 1
    j = 0
    while j < vector_end:
        weights.unsafe_store(j, weights.unsafe_load[width=W](j) / total)
        j += W
    while j < count:
        weights[unsafe_offset=j] /= total
        j += 1


@export("mho_simd_width")
def mho_simd_width() abi("C") -> Int:
    """Float64 lane count of the compiled density kernels (tests size their tails by it)."""
    return W


@export("mho_parallel_workers")
def mho_parallel_workers(n: Int, k: Int, requested: Int) abi("C") -> Int:
    """Workers the density kernels would use for this size (1 means serial)."""
    return choose_workers(n, k, requested)


@export("mho_gmm1_lpdf")
def mho_gmm1_lpdf(
    samples_addr: Int,
    n: Int,
    weights_addr: Int,
    mus_addr: Int,
    sigmas_addr: Int,
    k: Int,
    low: Float64,
    high: Float64,
    has_low: Int,
    has_high: Int,
    q: Float64,
    quantized: Int,
    result_addr: Int,
    workers: Int,
) abi("C") -> Int:
    if n < 0 or k <= 0 or weights_addr == 0 or mus_addr == 0 or sigmas_addr == 0:
        return 0
    if n > 0 and (samples_addr == 0 or result_addr == 0):
        return 0
    gmm_lpdf(
        fp(samples_addr),
        n,
        fp(weights_addr),
        fp(mus_addr),
        fp(sigmas_addr),
        k,
        low,
        high,
        has_low != 0,
        has_high != 0,
        q,
        quantized != 0,
        fp(result_addr),
        choose_workers(n, k, workers),
    )
    return 1


@export("mho_lgmm1_lpdf")
def mho_lgmm1_lpdf(
    samples_addr: Int,
    n: Int,
    weights_addr: Int,
    mus_addr: Int,
    sigmas_addr: Int,
    k: Int,
    low: Float64,
    high: Float64,
    has_low: Int,
    has_high: Int,
    q: Float64,
    quantized: Int,
    result_addr: Int,
    workers: Int,
) abi("C") -> Int:
    if n < 0 or k <= 0 or weights_addr == 0 or mus_addr == 0 or sigmas_addr == 0:
        return 0
    if n > 0 and (samples_addr == 0 or result_addr == 0):
        return 0
    lgmm_lpdf(
        fp(samples_addr),
        n,
        fp(weights_addr),
        fp(mus_addr),
        fp(sigmas_addr),
        k,
        low,
        high,
        has_low != 0,
        has_high != 0,
        q,
        quantized != 0,
        fp(result_addr),
        choose_workers(n, k, workers),
    )
    return 1


@export("mho_gmm1_lpdf_gpu")
def mho_gmm1_lpdf_gpu(
    samples_addr: Int,
    n: Int,
    log_coefficients_addr: Int,
    mus_addr: Int,
    sigmas_addr: Int,
    k: Int,
    log_space: Int,
    result_addr: Int,
) abi("C") -> Int:
    if n <= 0 or k <= 0:
        return 0
    if (
        samples_addr == 0
        or log_coefficients_addr == 0
        or mus_addr == 0
        or sigmas_addr == 0
        or result_addr == 0
    ):
        return 0
    try:
        gmm_lpdf_gpu(
            fp(samples_addr),
            n,
            fp(log_coefficients_addr),
            fp(mus_addr),
            fp(sigmas_addr),
            k,
            log_space,
            fp(result_addr),
        )
        return 1
    except:
        return 0


@export("mho_adaptive_parzen_normal")
def mho_adaptive_parzen_normal(
    observations_addr: Int,
    n: Int,
    prior_weight: Float64,
    prior_mu: Float64,
    prior_sigma: Float64,
    lf: Int,
    weights_addr: Int,
    mus_addr: Int,
    sigmas_addr: Int,
    order_addr: Int,
) abi("C") -> Int:
    if n < 0 or weights_addr == 0 or mus_addr == 0 or sigmas_addr == 0:
        return 0
    if n > 0 and (observations_addr == 0 or order_addr == 0):
        return 0
    adaptive_parzen(
        fp(observations_addr),
        n,
        prior_weight,
        prior_mu,
        prior_sigma,
        lf,
        fp(weights_addr),
        fp(mus_addr),
        fp(sigmas_addr),
        ip(order_addr),
    )
    return 1


@export("mho_categorical_lpdf")
def mho_categorical_lpdf(
    samples_addr: Int,
    n: Int,
    log_probabilities_addr: Int,
    category_count: Int,
    result_addr: Int,
) abi("C") -> Int:
    if n < 0 or category_count <= 0 or log_probabilities_addr == 0:
        return 0
    if n > 0 and (samples_addr == 0 or result_addr == 0):
        return 0
    var samples = ip(samples_addr)
    var log_probabilities = fp(log_probabilities_addr)
    var result = fp(result_addr)
    for i in range(n):
        if samples[unsafe_offset=i] < 0 or samples[unsafe_offset=i] >= Int64(category_count):
            return 0
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vector_end = (n // W) * W
    while i < vector_end:
        var indices = samples.unsafe_load[width=W](i)
        result.unsafe_store(i, log_probabilities.unsafe_gather(indices))
        i += W
    while i < n:
        result[unsafe_offset=i] = log_probabilities[unsafe_offset=Int(samples[unsafe_offset=i])]
        i += 1
    return 1


@export("mho_anneal_bounds")
def mho_anneal_bounds(
    centers_addr: Int,
    n: Int,
    low: Float64,
    high: Float64,
    shrinking: Float64,
    lower_addr: Int,
    upper_addr: Int,
) abi("C") -> Int:
    if n < 0:
        return 0
    if n > 0 and (centers_addr == 0 or lower_addr == 0 or upper_addr == 0):
        return 0
    var centers = fp(centers_addr)
    var lower = fp(lower_addr)
    var upper = fp(upper_addr)
    var half = 0.5 * (high - low) * shrinking
    var min_center = low + half
    var max_center = high - half

    comptime W = simd_width_of[DType.float64]()
    var min_center_vec = SIMD[DType.float64, W](min_center)
    var max_center_vec = SIMD[DType.float64, W](max_center)
    var i = 0
    var vector_end = (n // W) * W
    while i < vector_end:
        var center = centers.unsafe_load[width=W](i)
        center = max(min_center_vec, min(max_center_vec, center))
        lower.unsafe_store(i, center - half)
        upper.unsafe_store(i, center + half)
        i += W
    while i < n:
        var center = max(min_center, min(max_center, centers[unsafe_offset=i]))
        lower[unsafe_offset=i] = center - half
        upper[unsafe_offset=i] = center + half
        i += 1
    return 1
