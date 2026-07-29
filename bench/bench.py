"""Locked benchmarks against Hyperopt 0.2.7 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojohyperopt as mh  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import hyperopt as upstream  # noqa: E402


def machine() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def timeit(function, repeat=3):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


class NumPy2AnnealingAlgo(upstream.anneal.AnnealingAlgo):
    def choose_ltv(self, label, size):
        tids = self.node_tids[label]
        vals = self.node_vals[label]
        losses = [self.tid_losses_dct[tid] for tid in tids]
        if size == 1:
            tid_set = set(tids)
            for tid in self.best_tids:
                if tid in tid_set:
                    index = tids.index(tid)
                    return losses[index], tid, vals[index]
        good_index = self.rng.geometric(1.0 / self.avg_best_idx, size=size) - 1
        good_index = np.clip(good_index, 0, len(tids) - 1).astype("int32")
        picks = np.argsort(losses)[good_index]
        picked_loss = np.asarray(losses)[picks]
        picked_tids = np.asarray(tids)[picks]
        picked_vals = np.asarray(vals)[picks]
        if size == 1:
            self.best_tids.append(int(picked_tids.item()))
        return picked_loss, picked_tids, picked_vals


def upstream_anneal(new_ids, domain, trials, seed, *args, **kwargs):
    (new_id,) = new_ids
    return NumPy2AnnealingAlgo(domain, trials, seed, *args, **kwargs)(new_id)


CASES = []


def case(name):
    def register(function):
        CASES.append((name, function))
        return function

    return register


@case("adaptive Parzen, 20k observations")
def parzen_case():
    observations = np.random.default_rng(1).normal(size=20_000)
    return (
        lambda: mh.tpe.adaptive_parzen_normal(observations, 1.0, 0.0, 4.0),
        lambda: upstream.tpe.adaptive_parzen_normal(
            observations, 1.0, 0.0, 4.0
        ),
        5,
    )


@case("GMM log-density, 200k x 48")
def gmm_case():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=200_000)
    mus = np.linspace(-3, 3, 48)
    sigmas = np.linspace(0.15, 1.2, 48)
    weights = np.full(48, 1.0 / 48)
    return (
        lambda: mh.tpe.GMM1_lpdf(samples, weights, mus, sigmas),
        lambda: upstream.tpe.GMM1_lpdf(samples, weights, mus, sigmas),
        3,
    )


@case("GMM log-density GPU, 200k x 48")
def gmm_gpu_case():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=200_000)
    mus = np.linspace(-3, 3, 48)
    sigmas = np.linspace(0.15, 1.2, 48)
    weights = np.full(48, 1.0 / 48)
    return (
        lambda: mh.tpe.GMM1_lpdf(
            samples, weights, mus, sigmas, device="gpu"
        ),
        lambda: upstream.tpe.GMM1_lpdf(samples, weights, mus, sigmas),
        3,
    )


@case("quantized GMM log-mass, 100k x 32")
def quantized_gmm_case():
    rng = np.random.default_rng(3)
    samples = np.round(rng.uniform(-4, 4, size=100_000) * 4) / 4
    mus = np.linspace(-4, 4, 32)
    sigmas = np.linspace(0.2, 1.0, 32)
    weights = np.full(32, 1.0 / 32)
    return (
        lambda: mh.tpe.GMM1_lpdf(
            samples, weights, mus, sigmas, low=-5, high=5, q=0.25
        ),
        lambda: upstream.tpe.GMM1_lpdf(
            samples, weights, mus, sigmas, low=-5, high=5, q=0.25
        ),
        3,
    )


@case("log-GMM log-density, 200k x 48")
def log_gmm_case():
    rng = np.random.default_rng(4)
    samples = np.exp(rng.normal(size=200_000))
    mus = np.linspace(-3, 3, 48)
    sigmas = np.linspace(0.15, 1.2, 48)
    weights = np.full(48, 1.0 / 48)
    return (
        lambda: mh.tpe.LGMM1_lpdf(samples, weights, mus, sigmas),
        lambda: upstream.tpe.LGMM1_lpdf(samples, weights, mus, sigmas),
        3,
    )


@case("categorical log-density, 5M")
def categorical_case():
    rng = np.random.default_rng(5)
    samples = rng.integers(0, 64, size=5_000_000, dtype=np.int64)
    probabilities = rng.random(64)
    probabilities /= probabilities.sum()
    return (
        lambda: mh.tpe.categorical_lpdf(samples, probabilities),
        lambda: upstream.tpe.categorical_lpdf(samples, probabilities),
        3,
    )


@case("annealing bounds, 5M centers")
def bounds_case():
    centers = np.random.default_rng(6).normal(size=5_000_000)
    low, high, shrinking = -5.0, 7.0, 0.2
    half = 0.5 * (high - low) * shrinking
    return (
        lambda: mh.anneal.anneal_bounds(centers, low, high, shrinking),
        lambda: (
            np.clip(centers, low + half, high - half) - half,
            np.clip(centers, low + half, high - half) + half,
        ),
        3,
    )


def objective(point):
    return (point["x"] - 1.25) ** 2 + (np.log(point["y"]) + 0.5) ** 2


def run_ours(algorithm, evaluations):
    space = {
        "x": mh.hp.uniform("x", -5, 5),
        "y": mh.hp.loguniform("y", -3, 2),
    }
    trials = mh.Trials()
    mh.fmin(
        objective,
        space,
        algo=algorithm,
        max_evals=evaluations,
        trials=trials,
        rstate=np.random.default_rng(7),
        verbose=False,
    )
    return trials.best_trial["result"]["loss"]


def run_upstream(algorithm, evaluations):
    space = {
        "x": upstream.hp.uniform("x", -5, 5),
        "y": upstream.hp.loguniform("y", -3, 2),
    }
    trials = upstream.Trials()
    upstream.fmin(
        objective,
        space,
        algo=algorithm,
        max_evals=evaluations,
        trials=trials,
        rstate=np.random.default_rng(7),
        verbose=False,
        show_progressbar=False,
    )
    return trials.best_trial["result"]["loss"]


@case("TPE fmin, 200 cheap evaluations")
def tpe_fmin_case():
    return (
        lambda: run_ours(mh.tpe.suggest, 200),
        lambda: run_upstream(upstream.tpe.suggest, 200),
        2,
    )


@case("anneal fmin, 500 cheap evaluations")
def anneal_fmin_case():
    return (
        lambda: run_ours(mh.anneal.suggest, 500),
        lambda: run_upstream(upstream_anneal, 500),
        2,
    )


def main():
    mh.build()
    print(f"Machine: {machine()}; {platform.platform()}")
    print()
    print("| benchmark | mojo-hyperopt | hyperopt 0.2.7 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, prepare in CASES:
        ours, theirs, repeat = prepare()
        ours()
        their_time, their_result = timeit(theirs, repeat=repeat)
        our_time, our_result = timeit(ours, repeat=repeat)
        if isinstance(our_result, np.ndarray):
            if not np.allclose(our_result, their_result, rtol=1e-6, atol=3e-7):
                raise AssertionError(f"benchmark outputs differ for {name}")
        speedup = their_time / our_time
        print(
            f"| {name} | {our_time * 1e3:.2f} ms | "
            f"{their_time * 1e3:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
