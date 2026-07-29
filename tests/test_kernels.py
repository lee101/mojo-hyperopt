import warnings

import numpy as np
import pytest

from mojohyperopt import anneal, tpe

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from hyperopt import tpe as upstream_tpe


@pytest.mark.parametrize("n", [0, 1, 2, 10, 25, 26, 100])
def test_adaptive_parzen_matches_upstream(n):
    observations = np.random.default_rng(n).normal(size=n)
    ours = tpe.adaptive_parzen_normal(
        observations, prior_weight=1.3, prior_mu=0.2, prior_sigma=4.0
    )
    theirs = upstream_tpe.adaptive_parzen_normal(
        observations, prior_weight=1.3, prior_mu=0.2, prior_sigma=4.0
    )
    for actual, expected in zip(ours, theirs):
        assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_adaptive_parzen_rejects_non_vector_observations():
    with pytest.raises(TypeError, match="vector"):
        tpe.adaptive_parzen_normal([[1.0, 2.0]], 1.0, 0.0, 1.0)


def test_adaptive_parzen_simd_tail_matches_upstream():
    observations = np.array([3.0, -1.0, 0.5, 8.0, -4.0, 2.0])
    ours = tpe.adaptive_parzen_normal(observations, 1.3, 0.2, 4.0)
    theirs = upstream_tpe.adaptive_parzen_normal(
        observations, 1.3, 0.2, 4.0
    )
    for actual, expected in zip(ours, theirs):
        assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [0, 1, 24, 25, 26, 100])
def test_linear_forgetting_matches_upstream(n):
    assert np.allclose(
        tpe.linear_forgetting_weights(n, 25),
        upstream_tpe.linear_forgetting_weights(n, 25),
    )


@pytest.mark.parametrize(
    ("low", "high", "q"),
    [(None, None, None), (-2.0, 3.0, None), (-2.0, 3.0, 0.25)],
)
def test_gmm_lpdf_matches_upstream(low, high, q):
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(17, 5))
    weights = np.array([0.2, 0.3, 0.5])
    mus = np.array([-1.0, 0.5, 2.0])
    sigmas = np.array([0.4, 1.0, 0.2])
    actual = tpe.GMM1_lpdf(samples, weights, mus, sigmas, low, high, q)
    expected = upstream_tpe.GMM1_lpdf(
        samples, weights, mus, sigmas, low, high, q
    )
    assert actual.shape == samples.shape
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize(
    ("low", "high", "q"),
    [(None, None, None), (-2.0, 3.0, None), (-2.0, 3.0, 0.25)],
)
def test_lgmm_lpdf_matches_upstream(low, high, q):
    rng = np.random.default_rng(9)
    samples = np.exp(rng.normal(size=(11, 7)))
    weights = np.array([0.2, 0.3, 0.5])
    mus = np.array([-1.0, 0.5, 2.0])
    sigmas = np.array([0.4, 1.0, 0.2])
    actual = tpe.LGMM1_lpdf(samples, weights, mus, sigmas, low, high, q)
    expected = upstream_tpe.LGMM1_lpdf(
        samples, weights, mus, sigmas, low, high, q
    )
    assert actual.shape == samples.shape
    assert np.allclose(actual, expected, rtol=1e-6, atol=3e-7)


def test_categorical_lpdf_matches_upstream():
    samples = np.array([0, 3, 1, 1, 2, 0], dtype=np.int64)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(
        tpe.categorical_lpdf(samples, probabilities),
        upstream_tpe.categorical_lpdf(samples, probabilities),
        rtol=1e-8,
        atol=1e-8,
    )


def test_categorical_lpdf_simd_tail_matches_upstream():
    samples = np.array([0, 3, 1, 1, 2, 0, 3], dtype=np.int64)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(
        tpe.categorical_lpdf(samples, probabilities),
        upstream_tpe.categorical_lpdf(samples, probabilities),
    )


@pytest.mark.parametrize("function", [tpe.GMM1_lpdf, tpe.LGMM1_lpdf])
def test_gpu_lpdf_matches_cpu_when_available(function):
    if not tpe._gpu_memory_available(10_003, 7):
        pytest.skip("GPU unavailable or has less than 4000 MiB free")
    rng = np.random.default_rng(42)
    samples = rng.normal(size=10_003)
    if function is tpe.LGMM1_lpdf:
        samples = np.exp(samples)
    weights = np.full(7, 1.0 / 7)
    mus = np.linspace(-2, 2, 7)
    sigmas = np.linspace(0.2, 1.0, 7)
    cpu = function(samples, weights, mus, sigmas)
    gpu = function(samples, weights, mus, sigmas, device="gpu")
    assert np.allclose(gpu, cpu, rtol=1e-6, atol=3e-7)


def test_gpu_request_silently_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(tpe, "_gpu_memory_available", lambda n, k: False)
    samples = np.array([-1.0, 0.0, 1.0])
    weights = np.array([0.4, 0.6])
    mus = np.array([-0.5, 0.5])
    sigmas = np.array([0.8, 1.2])
    assert np.array_equal(
        tpe.GMM1_lpdf(samples, weights, mus, sigmas, device="gpu"),
        tpe.GMM1_lpdf(samples, weights, mus, sigmas),
    )


def test_lpdf_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        tpe.GMM1_lpdf([0.0], [1.0], [0.0], [1.0], device="other")


def test_split_trials_matches_upstream_with_duplicate_losses():
    observation_ids = np.array([2, 4, 6, 8, 10, 12])
    observations = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    loss_ids = np.array([2, 4, 6, 8, 10, 12])
    losses = np.array([1.0, 0.0, 0.0, 4.0, 2.0, 3.0])
    actual = tpe.ap_split_trials(
        observation_ids, observations, loss_ids, losses, 0.5
    )
    expected = upstream_tpe.ap_split_trials(
        observation_ids, observations, loss_ids, losses, 0.5
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_annealing_bounds_match_upstream_formula():
    centers = np.array([-20.0, -1.0, 0.0, 3.0, 20.0])
    low, high, shrinking = -5.0, 7.0, 0.25
    lower, upper = anneal.anneal_bounds(centers, low, high, shrinking)
    half = 0.5 * (high - low) * shrinking
    clipped = np.clip(centers, low + half, high - half)
    assert np.array_equal(lower, clipped - half)
    assert np.array_equal(upper, clipped + half)


@pytest.mark.parametrize("n", [7, 1_000_003])
def test_annealing_bounds_simd_tail_and_parallel_threshold(n):
    centers = np.random.default_rng(n).normal(size=n)
    low, high, shrinking = -5.0, 7.0, 0.2
    lower, upper = anneal.anneal_bounds(centers, low, high, shrinking)
    half = 0.5 * (high - low) * shrinking
    clipped = np.clip(centers, low + half, high - half)
    assert np.array_equal(lower, clipped - half)
    assert np.array_equal(upper, clipped + half)


@pytest.mark.parametrize(
    ("function", "args", "error"),
    [
        (tpe.GMM1_lpdf, ([0.0], [], [], []), "at least one"),
        (tpe.GMM1_lpdf, ([0.0], [1.0], [0.0], [0.0]), "sigmas"),
        (tpe.LGMM1_lpdf, ([0.0], [1.0], [0.0], [1.0]), "positive"),
        (tpe.categorical_lpdf, ([0.5], [0.5, 0.5]), "integer dtype"),
        (tpe.categorical_lpdf, ([-1], [0.5, 0.5]), "outside"),
        (tpe.categorical_lpdf, ([2], [0.5, 0.5]), "outside"),
    ],
)
def test_ffi_inputs_are_rejected_before_unsafe_access(function, args, error):
    with pytest.raises((TypeError, ValueError), match=error):
        function(*args)


def test_noncontiguous_inputs_are_copied_for_ffi():
    samples = np.arange(20.0)[::2]
    actual = tpe.GMM1_lpdf(samples, [1.0], [0.0], [1.0])
    expected = upstream_tpe.GMM1_lpdf(samples, [1.0], [0.0], [1.0])
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 31, 32, 33])
def test_categorical_simd_tails(n):
    samples = np.arange(n, dtype=np.int64) % 3
    probabilities = np.array([0.2, 0.3, 0.5])
    assert np.array_equal(
        tpe.categorical_lpdf(samples, probabilities),
        np.log(probabilities[samples]),
    )
