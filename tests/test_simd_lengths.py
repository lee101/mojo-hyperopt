"""SIMD tail coverage for every vectorised kernel.

Each vectorised path is driven at the lengths that pick out a different
remainder against the compiled lane count W: none, one element, W-1, exactly
W, W+1 (one full block and a one-element tail), four full blocks, and two large
odd lengths. Those reach the scalar tail loops, the single-block body, the
four-accumulator body and its tail.
"""

import warnings

import numpy as np
import pytest

from mojohyperopt import anneal, tpe

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from hyperopt import tpe as upstream_tpe

W = tpe.simd_width()
LENGTHS = [0, 1, W - 1, W, W + 1, 4 * W, 4 * W + 1, 97, 100_001]


def _components(k, seed):
    rng = np.random.default_rng(seed)
    weights = rng.random(k)
    weights /= weights.sum()
    return weights, rng.normal(size=k), rng.uniform(0.1, 1.5, size=k)


def test_simd_width_is_a_power_of_two():
    assert W >= 1 and W & (W - 1) == 0


@pytest.mark.parametrize("k", [1, W - 1, W, W + 1, 3 * W, 48])
@pytest.mark.parametrize("n", LENGTHS)
def test_gmm_simd_lengths(n, k):
    samples = np.random.default_rng(n * 100 + k).normal(size=n)
    weights, mus, sigmas = _components(max(k, 1), k)
    actual = tpe.GMM1_lpdf(samples, weights, mus, sigmas)
    expected = upstream_tpe.GMM1_lpdf(samples, weights, mus, sigmas)
    assert actual.shape == samples.shape
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize("k", [1, W - 1, W, W + 1, 3 * W, 48])
@pytest.mark.parametrize("n", LENGTHS)
def test_lgmm_simd_lengths(n, k):
    samples = np.exp(np.random.default_rng(n * 100 + k).normal(size=n))
    weights, mus, sigmas = _components(max(k, 1), k)
    actual = tpe.LGMM1_lpdf(samples, weights, mus, sigmas)
    expected = upstream_tpe.LGMM1_lpdf(samples, weights, mus, sigmas)
    assert actual.shape == samples.shape
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize("n", LENGTHS)
def test_quantized_gmm_line_integral_lengths(n):
    samples = np.round(np.random.default_rng(n).uniform(-3, 3, size=n) * 4) / 4
    weights, mus, sigmas = _components(9, 9)
    actual = tpe.GMM1_lpdf(samples, weights, mus, sigmas, low=-5.0, high=5.0, q=0.25)
    expected = upstream_tpe.GMM1_lpdf(
        samples, weights, mus, sigmas, low=-5.0, high=5.0, q=0.25
    )
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize("n", LENGTHS)
def test_quantized_lgmm_line_integral_lengths(n):
    samples = np.exp(np.random.default_rng(n).uniform(-2, 2, size=n))
    weights, mus, sigmas = _components(9, 19)
    actual = tpe.LGMM1_lpdf(samples, weights, mus, sigmas, low=-3.0, high=3.0, q=0.5)
    expected = upstream_tpe.LGMM1_lpdf(
        samples, weights, mus, sigmas, low=-3.0, high=3.0, q=0.5
    )
    assert actual.shape == samples.shape
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize("n", LENGTHS)
def test_categorical_simd_lengths(n):
    samples = np.arange(n, dtype=np.int64) % 5
    probabilities = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    actual = tpe.categorical_lpdf(samples, probabilities)
    assert np.array_equal(actual, np.log(probabilities[samples]))


@pytest.mark.parametrize("n", LENGTHS)
def test_adaptive_parzen_simd_lengths(n):
    observations = np.random.default_rng(n).normal(size=n)
    ours = tpe.adaptive_parzen_normal(observations, 1.3, 0.2, 4.0)
    theirs = upstream_tpe.adaptive_parzen_normal(observations, 1.3, 0.2, 4.0)
    for actual, expected in zip(ours, theirs):
        assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", LENGTHS)
def test_anneal_bounds_simd_lengths(n):
    centers = np.random.default_rng(n).normal(size=n)
    low, high, shrinking = -5.0, 7.0, 0.2
    lower, upper = anneal.anneal_bounds(centers, low, high, shrinking)
    half = 0.5 * (high - low) * shrinking
    clipped = np.clip(centers, low + half, high - half)
    assert np.array_equal(lower, clipped - half)
    assert np.array_equal(upper, clipped + half)
