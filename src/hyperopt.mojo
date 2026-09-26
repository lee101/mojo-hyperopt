"""Numerical kernels for TPE density estimation and annealing proposals."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import erf, exp, log, sqrt
from std.sys import simd_width_of


comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime EPS = 1.0e-12
comptime SQRT_2 = 1.4142135623730950488
comptime SQRT_2PI = 2.5066282746310005024


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def normal_cdf(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    var denom = SQRT_2 * sigma
    if denom < EPS:
        denom = EPS
    return 0.5 * (1.0 + erf((x - mu) / denom))


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
):
    var p_accept = acceptance(
        weights, mus, sigmas, k, low, high, has_low, has_high
    )
    if not quantized:
        for i in range(n):
            var max_term = -1.0e300
            for j in range(k):
                var sigma = sigmas[unsafe_offset=j]
                if sigma < EPS:
                    sigma = EPS
                var z = (samples[unsafe_offset=i] - mus[unsafe_offset=j]) / sigma
                var term = -0.5 * z * z + log(weights[unsafe_offset=j] / (SQRT_2PI * sigma))
                if term > max_term:
                    max_term = term
            var total = Float64(0.0)
            for j in range(k):
                var sigma = sigmas[unsafe_offset=j]
                if sigma < EPS:
                    sigma = EPS
                var z = (samples[unsafe_offset=i] - mus[unsafe_offset=j]) / sigma
                var term = -0.5 * z * z + log(weights[unsafe_offset=j] / (SQRT_2PI * sigma))
                total += exp(term - max_term)
            result[unsafe_offset=i] = log(total) + max_term - log(p_accept)
        return

    for i in range(n):
        var lower = samples[unsafe_offset=i] - 0.5 * q
        var upper = samples[unsafe_offset=i] + 0.5 * q
        if has_low and lower < low:
            lower = low
        if has_high and upper > high:
            upper = high
        var probability = Float64(0.0)
        for j in range(k):
            probability += weights[unsafe_offset=j] * (
                normal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
                - normal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
            )
        result[unsafe_offset=i] = log(probability) - log(p_accept)


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
):
    var p_accept = acceptance(
        weights, mus, sigmas, k, low, high, has_low, has_high
    )
    if not quantized:
        for i in range(n):
            var max_term = -1.0e300
            for j in range(k):
                var sigma = sigmas[unsafe_offset=j]
                if sigma < EPS:
                    sigma = EPS
                var z = (log(samples[unsafe_offset=i]) - mus[unsafe_offset=j]) / sigma
                var term = (
                    -0.5 * z * z
                    + log(weights[unsafe_offset=j])
                    - log(SQRT_2PI * sigma * samples[unsafe_offset=i])
                )
                if term > max_term:
                    max_term = term
            var total = Float64(0.0)
            for j in range(k):
                var sigma = sigmas[unsafe_offset=j]
                if sigma < EPS:
                    sigma = EPS
                var z = (log(samples[unsafe_offset=i]) - mus[unsafe_offset=j]) / sigma
                var term = (
                    -0.5 * z * z
                    + log(weights[unsafe_offset=j])
                    - log(SQRT_2PI * sigma * samples[unsafe_offset=i])
                )
                total += exp(term - max_term)
            result[unsafe_offset=i] = log(total) + max_term
        return

    for i in range(n):
        var lower = samples[unsafe_offset=i] - 0.5 * q
        var upper = samples[unsafe_offset=i] + 0.5 * q
        if lower < 0.0:
            lower = 0.0
        if has_low:
            var exp_low = exp(low)
            if lower < exp_low:
                lower = exp_low
        if has_high:
            var exp_high = exp(high)
            if upper > exp_high:
                upper = exp_high
        var probability = Float64(0.0)
        for j in range(k):
            probability += weights[unsafe_offset=j] * (
                lognormal_cdf(upper, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
                - lognormal_cdf(lower, mus[unsafe_offset=j], sigmas[unsafe_offset=j])
            )
        result[unsafe_offset=i] = log(probability) - log(p_accept)


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
