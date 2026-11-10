"""Serial vs threaded density kernels: identical output, and a measured threshold.

The parallel path exists only above a size where it was measured to win, so the
tests here check both halves of that contract: the results must not depend on
the worker count (bitwise, not within a tolerance), and sizes below the
threshold must still be handed to a single thread.
"""

import numpy as np
import pytest

from mojohyperopt import _lib, tpe

LIB = _lib.lib()
THRESHOLD_SAMPLES = 16384


def _mixture(k, seed):
    rng = np.random.default_rng(seed)
    weights = rng.random(k)
    weights /= weights.sum()
    return weights, rng.normal(size=k), rng.uniform(0.1, 1.5, size=k)


def _call(symbol, samples, weights, mus, sigmas, workers):
    sample_array = _lib.f64(samples).reshape(-1)
    result = np.empty(sample_array.size, dtype=np.float64)
    _lib.checked_call(
        symbol,
        _lib.addr(sample_array),
        sample_array.size,
        _lib.addr(_lib.f64(weights)),
        _lib.addr(_lib.f64(mus)),
        _lib.addr(_lib.f64(sigmas)),
        len(weights),
        0.0,
        0.0,
        0,
        0,
        0.0,
        0,
        _lib.addr(result),
        workers,
    )
    return result


@pytest.mark.parametrize("symbol", ["mho_gmm1_lpdf", "mho_lgmm1_lpdf"])
@pytest.mark.parametrize("workers", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("k", [1, 7, 48])
@pytest.mark.parametrize("n", [16383, 16384, 16385, 40_001])
def test_parallel_kernels_match_serial_bitwise(symbol, workers, k, n):
    weights, mus, sigmas = _mixture(k, seed=k)
    rng = np.random.default_rng(n)
    if symbol == "mho_lgmm1_lpdf":
        samples = rng.uniform(0.05, 4.0, size=n)
    else:
        samples = rng.normal(size=n)

    serial = _call(symbol, samples, weights, mus, sigmas, 1)
    threaded = _call(symbol, samples, weights, mus, sigmas, workers)
    assert np.isfinite(serial).all()
    # Same arithmetic per row and the same block split, so no tolerance needed:
    # this is what would fail if a worker ever shared a boundary row.
    assert np.array_equal(serial, threaded)


@pytest.mark.parametrize("symbol", ["mho_gmm1_lpdf", "mho_lgmm1_lpdf"])
def test_public_wrappers_ignore_worker_count(symbol):
    weights, mus, sigmas = _mixture(48, seed=1)
    rng = np.random.default_rng(7)
    if symbol == "mho_lgmm1_lpdf":
        samples = rng.uniform(0.05, 4.0, size=50_000)
    else:
        samples = rng.normal(size=50_000)
    function = tpe.GMM1_lpdf if symbol == "mho_gmm1_lpdf" else tpe.LGMM1_lpdf
    threaded = function(samples, weights, mus, sigmas)

    previous = tpe.PARALLEL_WORKERS
    tpe.PARALLEL_WORKERS = 1
    try:
        serial = function(samples, weights, mus, sigmas)
    finally:
        tpe.PARALLEL_WORKERS = previous
    assert np.array_equal(serial, threaded)


@pytest.mark.parametrize(
    "n,k,requested,expected",
    [
        (0, 96, 4, 1),
        (1, 48, 4, 1),
        (THRESHOLD_SAMPLES - 1, 48, 4, 1),
        (THRESHOLD_SAMPLES, 48, 4, 1),  # 16384 * 48 < MIN_WORK: narrow
        (THRESHOLD_SAMPLES, 96, 4, 1),  # size-eligible, but only 1 min-size block
        (2 * THRESHOLD_SAMPLES, 96, 1, 1),  # explicit serial
        (2 * THRESHOLD_SAMPLES, 96, 2, 2),  # prefixes of any larger split
        (2 * THRESHOLD_SAMPLES, 96, 4, 2),
        (2 * THRESHOLD_SAMPLES, 96, 64, 2),  # request never exceeds capacity here
        (4 * THRESHOLD_SAMPLES, 96, 4, 4),
        (4 * THRESHOLD_SAMPLES, 96, 64, 4),  # host cap is 4 however large
        (8 * THRESHOLD_SAMPLES, 96, 64, 4),
        (4 * THRESHOLD_SAMPLES, 96, 3, 3),
    ],
)
def test_worker_policy_matches_measured_threshold(n, k, requested, expected):
    assert LIB.mho_parallel_workers(n, k, requested) == expected


def test_callers_below_threshold_stay_serial(monkeypatch):
    calls = []
    original = tpe._lib.checked_call

    def record(symbol, *args):
        calls.append(symbol)
        return original(symbol, *args)

    monkeypatch.setattr(tpe._lib, "checked_call", record)
    weights, mus, sigmas = _mixture(48, seed=2)
    tpe.GMM1_lpdf(np.zeros(1000), weights, mus, sigmas)
    assert calls == ["mho_gmm1_lpdf"]
    # The kernel is told the caller's worker request and floors it itself; this
    # is the check that 1000 samples never reaches `parallelize`.
    assert LIB.mho_parallel_workers(1000, 48, tpe.PARALLEL_WORKERS) == 1
