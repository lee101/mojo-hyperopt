"""Tree-structured Parzen estimator search with Mojo density kernels."""

from __future__ import annotations

import math
import re
import subprocess
from typing import Any

import numpy as np

from . import _lib
from .base import STATUS_OK, trial_doc
from .hp import Distribution
from .space import prior_distribution

EPS = 1e-12
DEFAULT_LF = 25
_default_prior_weight = 1.0
_default_n_EI_candidates = 24
_default_gamma = 0.25
_default_n_startup_jobs = 20
_GPU_MIN_FREE_MIB = 4000
_GPU_MAX_ALLOCATION_BYTES = 2 * 1024**3


def _gpu_memory_available(n: int, k: int) -> bool:
    allocation_bytes = 8 * (2 * n + 3 * k)
    if allocation_bytes >= _GPU_MAX_ALLOCATION_BYTES:
        return False
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    free_values = [int(value) for value in re.findall(r"\d+", proc.stdout)]
    return bool(free_values) and free_values[0] >= _GPU_MIN_FREE_MIB


def _gpu_lpdf(
    samples: np.ndarray,
    weights: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    log_space: bool,
    result: np.ndarray,
) -> bool:
    if samples.size == 0 or not _gpu_memory_available(
        samples.size, weights.size
    ):
        return False
    safe_sigmas = np.maximum(sigmas, EPS)
    log_coefficients = np.log(weights / (math.sqrt(2.0 * math.pi) * safe_sigmas))
    status = _lib.lib().mho_gmm1_lpdf_gpu(
        _lib.addr(samples),
        samples.size,
        _lib.addr(log_coefficients),
        _lib.addr(mus),
        _lib.addr(sigmas),
        weights.size,
        int(log_space),
        _lib.addr(result),
    )
    if not status:
        raise RuntimeError("Mojo GPU density kernel failed")
    return True


def linear_forgetting_weights(N, LF):
    if N < 0 or LF <= 0:
        raise ValueError("require N >= 0 and LF > 0")
    if N == 0:
        return np.asarray([], dtype=np.float64)
    if N < LF:
        return np.ones(N)
    return np.concatenate(
        [np.linspace(1.0 / N, 1.0, num=N - LF), np.ones(LF)]
    )


def adaptive_parzen_normal(
    mus, prior_weight, prior_mu, prior_sigma, LF=DEFAULT_LF
):
    original = np.asarray(mus)
    if original.ndim != 1:
        raise TypeError("mus must be vector")
    observations = _lib.f64(original)
    if not np.isfinite(observations).all():
        raise ValueError("mus must be finite")
    if not np.isfinite([prior_weight, prior_mu, prior_sigma]).all():
        raise ValueError("prior parameters must be finite")
    if prior_weight <= 0 or prior_sigma <= 0:
        raise ValueError("prior_weight and prior_sigma must be positive")
    if LF <= 0:
        raise ValueError("LF must be positive")
    count = observations.size + 1
    weights = np.empty(count, dtype=np.float64)
    centers = np.empty(count, dtype=np.float64)
    sigmas = np.empty(count, dtype=np.float64)
    order = np.argsort(observations)
    _lib.checked_call(
        "mho_adaptive_parzen_normal",
        _lib.addr(observations),
        observations.size,
        float(prior_weight),
        float(prior_mu),
        float(prior_sigma),
        int(LF),
        _lib.addr(weights),
        _lib.addr(centers),
        _lib.addr(sigmas),
        _lib.addr(order),
    )
    return weights, centers, sigmas


def GMM1_lpdf(
    samples, weights, mus, sigmas, low=None, high=None, q=None, device="cpu"
):
    original = np.asarray(samples)
    sample_array = _lib.f64(original).reshape(-1)
    weight_array = _lib.f64(weights)
    center_array = _lib.f64(mus)
    sigma_array = _lib.f64(sigmas)
    if weight_array.ndim != 1 or center_array.ndim != 1 or sigma_array.ndim != 1:
        raise TypeError("weights, mus and sigmas must be vectors")
    if not (weight_array.size == center_array.size == sigma_array.size):
        raise ValueError("weights, mus and sigmas must have equal length")
    if weight_array.size == 0:
        raise ValueError("mixtures require at least one component")
    if not np.isfinite(sample_array).all():
        raise ValueError("samples must be finite")
    if not np.isfinite(weight_array).all() or np.any(weight_array <= 0):
        raise ValueError("weights must be finite and positive")
    if not np.isfinite(center_array).all():
        raise ValueError("mus must be finite")
    if not np.isfinite(sigma_array).all() or np.any(sigma_array <= 0):
        raise ValueError("sigmas must be finite and positive")
    if q is not None and (not np.isfinite(q) or q <= 0):
        raise ValueError("q must be positive and finite")
    if low is not None and not np.isfinite(low):
        raise ValueError("low must be finite")
    if high is not None and not np.isfinite(high):
        raise ValueError("high must be finite")
    if low is not None and high is not None and high <= low:
        raise ValueError("high must exceed low")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    result = np.empty(sample_array.size, dtype=np.float64)
    if (
        device == "gpu"
        and low is None
        and high is None
        and q is None
        and _gpu_lpdf(
            sample_array,
            weight_array,
            center_array,
            sigma_array,
            False,
            result,
        )
    ):
        return result.reshape(original.shape)
    _lib.checked_call(
        "mho_gmm1_lpdf",
        _lib.addr(sample_array),
        sample_array.size,
        _lib.addr(weight_array),
        _lib.addr(center_array),
        _lib.addr(sigma_array),
        weight_array.size,
        0.0 if low is None else float(low),
        0.0 if high is None else float(high),
        int(low is not None),
        int(high is not None),
        0.0 if q is None else float(q),
        int(q is not None),
        _lib.addr(result),
    )
    return result.reshape(original.shape)


def LGMM1_lpdf(
    samples, weights, mus, sigmas, low=None, high=None, q=None, device="cpu"
):
    original = np.asarray(samples)
    sample_array = _lib.f64(original).reshape(-1)
    weight_array = _lib.f64(weights)
    center_array = _lib.f64(mus)
    sigma_array = _lib.f64(sigmas)
    if weight_array.ndim != 1 or center_array.ndim != 1 or sigma_array.ndim != 1:
        raise TypeError("weights, mus and sigmas must be vectors")
    if not (weight_array.size == center_array.size == sigma_array.size):
        raise ValueError("weights, mus and sigmas must have equal length")
    if weight_array.size == 0:
        raise ValueError("mixtures require at least one component")
    if not np.isfinite(sample_array).all() or np.any(sample_array <= 0):
        raise ValueError("log-GMM samples must be finite and positive")
    if not np.isfinite(weight_array).all() or np.any(weight_array <= 0):
        raise ValueError("weights must be finite and positive")
    if not np.isfinite(center_array).all():
        raise ValueError("mus must be finite")
    if not np.isfinite(sigma_array).all() or np.any(sigma_array <= 0):
        raise ValueError("sigmas must be finite and positive")
    if q is not None and (not np.isfinite(q) or q <= 0):
        raise ValueError("q must be positive and finite")
    if low is not None and not np.isfinite(low):
        raise ValueError("low must be finite")
    if high is not None and not np.isfinite(high):
        raise ValueError("high must be finite")
    if low is not None and high is not None and high <= low:
        raise ValueError("high must exceed low")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    result = np.empty(sample_array.size, dtype=np.float64)
    if (
        device == "gpu"
        and low is None
        and high is None
        and q is None
        and _gpu_lpdf(
            sample_array,
            weight_array,
            center_array,
            sigma_array,
            True,
            result,
        )
    ):
        return result.reshape(original.shape)
    _lib.checked_call(
        "mho_lgmm1_lpdf",
        _lib.addr(sample_array),
        sample_array.size,
        _lib.addr(weight_array),
        _lib.addr(center_array),
        _lib.addr(sigma_array),
        weight_array.size,
        0.0 if low is None else float(low),
        0.0 if high is None else float(high),
        int(low is not None),
        int(high is not None),
        0.0 if q is None else float(q),
        int(q is not None),
        _lib.addr(result),
    )
    return result.reshape(original.shape)


def categorical_lpdf(sample, p):
    original = np.asarray(sample)
    if original.dtype.kind not in "iu":
        raise TypeError("categorical samples must have an integer dtype")
    samples = _lib.i64(original).reshape(-1)
    probabilities = _lib.f64(p).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("probabilities must not be empty")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")
    if samples.size and (samples.min() < 0 or samples.max() >= probabilities.size):
        raise ValueError("categorical sample is outside the probability table")
    log_probabilities = np.log(probabilities)
    result = np.empty(samples.size, dtype=np.float64)
    _lib.checked_call(
        "mho_categorical_lpdf",
        _lib.addr(samples),
        samples.size,
        _lib.addr(log_probabilities),
        probabilities.size,
        _lib.addr(result),
    )
    return result.reshape(original.shape)


def ap_split_trials(
    o_idxs, o_vals, l_idxs, l_vals, gamma, gamma_cap=DEFAULT_LF
):
    o_idxs, o_vals, l_idxs, l_vals = [
        np.asarray(value) for value in (o_idxs, o_vals, l_idxs, l_vals)
    ]
    n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
    order = np.argsort(l_vals)
    below_ids = set(l_idxs[order[:n_below]])
    above_ids = set(l_idxs[order[n_below:]])
    below = [value for index, value in zip(o_idxs, o_vals) if index in below_ids]
    above = [value for index, value in zip(o_idxs, o_vals) if index in above_ids]
    return np.asarray(below), np.asarray(above)


def _categorical_model(
    observations: np.ndarray, probabilities: np.ndarray, prior_weight: float
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.int64)
    weights = linear_forgetting_weights(len(observations), DEFAULT_LF)
    counts = np.bincount(
        observations, weights=weights, minlength=len(probabilities)
    ).astype(np.float64)
    pseudocounts = counts + len(probabilities) * prior_weight * probabilities
    return pseudocounts / pseudocounts.sum()


def _numeric_model(
    node: Distribution, observations, prior_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)
    name = node.name
    if name in ("uniform", "quniform", "loguniform", "qloguniform"):
        low, high = node.parameters[:2]
        prior_mu = 0.5 * (low + high)
        prior_sigma = high - low
    else:
        prior_mu, prior_sigma = node.parameters[:2]
    if name in ("loguniform", "qloguniform", "lognormal", "qlognormal"):
        floor_value = max(EPS, math.exp(node.parameters[0])) if "uniform" in name else EPS
        values = np.log(np.maximum(values, floor_value))
    return adaptive_parzen_normal(
        values, prior_weight, prior_mu, prior_sigma, LF=DEFAULT_LF
    )


def _sample_mixture(
    node: Distribution,
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    weights, mus, sigmas = model
    name = node.name
    bounded = name in ("uniform", "quniform", "loguniform", "qloguniform")
    low, high = node.parameters[:2] if bounded else (None, None)
    accepted: list[float] = []
    while len(accepted) < size:
        remaining = size - len(accepted)
        active = np.argmax(rng.multinomial(1, weights, size=remaining), axis=1)
        draws = rng.normal(mus[active], sigmas[active])
        if bounded:
            draws = draws[(draws >= low) & (draws < high)]
        accepted.extend(float(value) for value in draws)
    values = np.asarray(accepted[:size])
    if name in ("loguniform", "qloguniform", "lognormal", "qlognormal"):
        values = np.exp(values)
    if name in ("quniform", "qloguniform", "qnormal", "qlognormal"):
        q = node.parameters[2]
        values = np.round(values / q) * q
    return values


def _numeric_lpdf(node: Distribution, samples, model):
    weights, mus, sigmas = model
    name = node.name
    bounded = name in ("uniform", "quniform", "loguniform", "qloguniform")
    low, high = node.parameters[:2] if bounded else (None, None)
    q = (
        node.parameters[2]
        if name in ("quniform", "qloguniform", "qnormal", "qlognormal")
        else None
    )
    if name in ("loguniform", "qloguniform", "lognormal", "qlognormal"):
        return LGMM1_lpdf(samples, weights, mus, sigmas, low=low, high=high, q=q)
    return GMM1_lpdf(samples, weights, mus, sigmas, low=low, high=high, q=q)


def _split_history(trials, gamma):
    docs = list(trials.trials)
    tids = np.asarray([trial["tid"] for trial in docs], dtype=np.int64)
    losses = np.asarray(
        [
            float(trial["result"].get("loss", float("inf")))
            if trial["result"].get("loss") is not None
            else float("inf")
            for trial in docs
        ],
        dtype=np.float64,
    )
    n_below = min(int(np.ceil(gamma * np.sqrt(len(docs)))), DEFAULT_LF)
    order = np.argsort(losses)
    below_ids = set(int(tid) for tid in tids[order[:n_below]])
    above_ids = set(int(tid) for tid in tids[order[n_below:]])
    below: dict[str, list[Any]] = {}
    above: dict[str, list[Any]] = {}
    for trial in docs:
        target = below if trial["tid"] in below_ids else above
        if trial["tid"] not in below_ids and trial["tid"] not in above_ids:
            continue
        for label, values in trial["misc"]["vals"].items():
            if values:
                target.setdefault(label, []).append(values[0])
    return below, above


def _posterior_sample(
    space,
    below: dict[str, list[Any]],
    above: dict[str, list[Any]],
    rng: np.random.Generator,
    count: int,
    prior_weight: float,
):
    assignments: list[dict[str, Any]] = [{} for _ in range(count)]
    scores = np.zeros(count, dtype=np.float64)

    def sample(node, indices: list[int]):
        if isinstance(node, Distribution):
            good_obs = below.get(node.label, [])
            bad_obs = above.get(node.label, [])
            if node.name in ("choice", "pchoice", "randint"):
                if node.name == "randint":
                    offset = int(node.parameters[0])
                    categories = int(node.parameters[1] - node.parameters[0])
                    prior = np.full(categories, 1.0 / categories)
                    good_values = np.asarray(good_obs, dtype=np.int64) - offset
                    bad_values = np.asarray(bad_obs, dtype=np.int64) - offset
                else:
                    offset = 0
                    categories = len(node.options)
                    prior = (
                        np.full(categories, 1.0 / categories)
                        if node.name == "choice"
                        else np.asarray(node.probabilities, dtype=np.float64)
                    )
                    good_values = np.asarray(good_obs, dtype=np.int64)
                    bad_values = np.asarray(bad_obs, dtype=np.int64)
                good_model = _categorical_model(good_values, prior, prior_weight)
                bad_model = _categorical_model(bad_values, prior, prior_weight)
                category_values = rng.choice(
                    categories, size=len(indices), p=good_model
                ).astype(np.int64)
                raw_values = category_values + offset
                scores[np.asarray(indices)] += categorical_lpdf(
                    category_values, good_model
                ) - categorical_lpdf(category_values, bad_model)
                for index, value in zip(indices, raw_values):
                    assignments[index][node.label] = int(value)
                if node.name in ("choice", "pchoice"):
                    results = [None] * len(indices)
                    for category in range(categories):
                        positions = [
                            position
                            for position, value in enumerate(category_values)
                            if value == category
                        ]
                        if not positions:
                            continue
                        child_indices = [indices[position] for position in positions]
                        child_values = sample(node.options[category], child_indices)
                        for position, value in zip(positions, child_values):
                            results[position] = value
                    return results
                return [
                    int(value) if node.cast_int else int(value) for value in raw_values
                ]

            good_model = _numeric_model(node, good_obs, prior_weight)
            bad_model = _numeric_model(node, bad_obs, prior_weight)
            values = _sample_mixture(node, good_model, rng, len(indices))
            scores[np.asarray(indices)] += _numeric_lpdf(
                node, values, good_model
            ) - _numeric_lpdf(node, values, bad_model)
            for index, value in zip(indices, values):
                assignments[index][node.label] = float(value)
            return [int(value) if node.cast_int else float(value) for value in values]
        if isinstance(node, dict):
            children = {key: sample(value, indices) for key, value in node.items()}
            return [
                {key: children[key][position] for key in children}
                for position in range(len(indices))
            ]
        if isinstance(node, list):
            children = [sample(value, indices) for value in node]
            return [
                [child[position] for child in children]
                for position in range(len(indices))
            ]
        if isinstance(node, tuple):
            children = [sample(value, indices) for value in node]
            return [
                tuple(child[position] for child in children)
                for position in range(len(indices))
            ]
        return [node] * len(indices)

    values = sample(space, list(range(count)))
    best = int(np.argmax(scores))
    return values[best], assignments[best]


def suggest(
    new_ids,
    domain,
    trials,
    seed,
    prior_weight=_default_prior_weight,
    n_startup_jobs=_default_n_startup_jobs,
    n_EI_candidates=_default_n_EI_candidates,
    gamma=_default_gamma,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    docs = []
    for new_id in new_ids:
        if len(trials.trials) + len(docs) < n_startup_jobs:
            raw: dict[str, Any] = {}

            def sample_prior_node(node):
                if isinstance(node, Distribution):
                    value = prior_distribution(node, rng, 1)[0]
                    if node.name in ("choice", "pchoice", "randint"):
                        value = int(value)
                    else:
                        value = float(value)
                    raw[node.label] = value
                    if node.name in ("choice", "pchoice"):
                        return sample_prior_node(node.options[int(value)])
                    return int(value) if node.cast_int else value
                if isinstance(node, dict):
                    return {key: sample_prior_node(value) for key, value in node.items()}
                if isinstance(node, list):
                    return [sample_prior_node(value) for value in node]
                if isinstance(node, tuple):
                    return tuple(sample_prior_node(value) for value in node)
                return node

            sample_prior_node(domain.space)
        else:
            below, above = _split_history(trials, gamma)
            _, raw = _posterior_sample(
                domain.space,
                below,
                above,
                rng,
                int(n_EI_candidates),
                float(prior_weight),
            )
        docs.append(trial_doc(int(new_id), raw))
    return docs
