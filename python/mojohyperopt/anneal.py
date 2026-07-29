"""Hyperopt-style annealing search over the supported recursive spaces."""

from __future__ import annotations

import concurrent.futures
import math
import os
from typing import Any

import numpy as np

from . import _lib
from .base import trial_doc
from .hp import Distribution
from .space import prior_distribution

_PARALLEL_THRESHOLD = 1_000_000
_PARALLEL_WORKERS = min(8, os.cpu_count() or 1)
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _bounds_call(center_array, lower, upper, low, high, shrinking):
    _lib.checked_call(
        "mho_anneal_bounds",
        _lib.addr(center_array),
        center_array.size,
        float(low),
        float(high),
        float(shrinking),
        _lib.addr(lower),
        _lib.addr(upper),
    )


def anneal_bounds(centers, low, high, shrinking):
    global _executor
    original = np.asarray(centers)
    center_array = _lib.f64(original).reshape(-1)
    low = float(low)
    high = float(high)
    shrinking = float(shrinking)
    if not np.isfinite([low, high, shrinking]).all():
        raise ValueError("bounds and shrinking must be finite")
    if high <= low:
        raise ValueError("high must exceed low")
    if not 0.0 <= shrinking <= 1.0:
        raise ValueError("shrinking must be between 0 and 1")
    lower = np.empty(center_array.size, dtype=np.float64)
    upper = np.empty(center_array.size, dtype=np.float64)
    if center_array.size < _PARALLEL_THRESHOLD or _PARALLEL_WORKERS == 1:
        _bounds_call(center_array, lower, upper, low, high, shrinking)
    else:
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_PARALLEL_WORKERS
            )
        chunk = (center_array.size + _PARALLEL_WORKERS - 1) // _PARALLEL_WORKERS
        futures = []
        for start in range(0, center_array.size, chunk):
            end = min(start + chunk, center_array.size)
            futures.append(
                _executor.submit(
                    _bounds_call,
                    center_array[start:end],
                    lower[start:end],
                    upper[start:end],
                    low,
                    high,
                    shrinking,
                )
            )
        for future in futures:
            future.result()
    return lower.reshape(original.shape), upper.reshape(original.shape)


class AnnealingAlgo:
    def __init__(self, domain, trials, seed, avg_best_idx=2.0, shrink_coef=0.1):
        if avg_best_idx < 1:
            raise ValueError("avg_best_idx must be at least 1")
        self.domain = domain
        self.trials = trials
        self.rng = np.random.default_rng(seed)
        self.avg_best_idx = float(avg_best_idx)
        self.shrink_coef = float(shrink_coef)
        self.best_tids: list[int] = []

    def _history(self, label):
        history = []
        for trial in self.trials.trials:
            values = trial["misc"]["vals"].get(label, [])
            if values:
                loss = trial["result"].get("loss")
                history.append(
                    (
                        float("inf") if loss is None else float(loss),
                        int(trial["tid"]),
                        values[0],
                    )
                )
        return history

    def shrinking(self, label):
        return 1.0 / (1.0 + len(self._history(label)) * self.shrink_coef)

    def _choose(self, label):
        history = self._history(label)
        by_tid = {tid: (loss, value) for loss, tid, value in history}
        for tid in self.best_tids:
            if tid in by_tid:
                loss, value = by_tid[tid]
                return loss, tid, value
        ordered = sorted(history, key=lambda item: item[0])
        rank = int(self.rng.geometric(1.0 / self.avg_best_idx) - 1)
        rank = min(rank, len(ordered) - 1)
        loss, tid, value = ordered[rank]
        self.best_tids.append(tid)
        return loss, tid, value

    def _sample_distribution(self, node: Distribution):
        history = self._history(node.label)
        if not history:
            value = prior_distribution(node, self.rng, 1)[0]
            return (
                int(value)
                if node.name in ("choice", "pchoice", "randint")
                else float(value)
            )

        _, _, previous = self._choose(node.label)
        shrinking = self.shrinking(node.label)
        name = node.name
        if name in ("choice", "pchoice", "randint"):
            if name == "randint":
                low, high = (int(value) for value in node.parameters)
                offset = low
                probabilities = np.full(high - low, 1.0 / (high - low))
            else:
                offset = 0
                probabilities = (
                    np.full(len(node.options), 1.0 / len(node.options))
                    if name == "choice"
                    else np.asarray(node.probabilities, dtype=np.float64)
                )
            counts = np.zeros(len(probabilities), dtype=np.float64)
            counts[int(previous) - offset] = 1.0
            proposal = (1.0 - shrinking) * counts + shrinking * probabilities
            return int(self.rng.choice(len(probabilities), p=proposal) + offset)

        if name in ("uniform", "quniform", "loguniform", "qloguniform"):
            low, high = node.parameters[:2]
            center = math.log(max(float(previous), 1e-16)) if "log" in name else float(previous)
            lower, upper = anneal_bounds([center], low, high, shrinking)
            latent = float(self.rng.uniform(lower[0], upper[0]))
            value = math.exp(latent) if "log" in name else latent
        elif name in ("normal", "qnormal"):
            value = float(
                self.rng.normal(float(previous), node.parameters[1] * shrinking)
            )
        elif name in ("lognormal", "qlognormal"):
            value = math.exp(
                float(
                    self.rng.normal(
                        math.log(1e-16 + float(previous)),
                        node.parameters[1] * shrinking,
                    )
                )
            )
        else:
            raise NotImplementedError(f"unsupported distribution {name!r}")

        if name in ("quniform", "qloguniform", "qnormal", "qlognormal"):
            q = node.parameters[2]
            value = float(np.round(value / q) * q)
        return value

    def sample(self):
        raw: dict[str, Any] = {}

        def visit(node):
            if isinstance(node, Distribution):
                value = self._sample_distribution(node)
                raw[node.label] = value
                if node.name in ("choice", "pchoice"):
                    return visit(node.options[int(value)])
                return int(value) if node.cast_int else value
            if isinstance(node, dict):
                return {key: visit(value) for key, value in node.items()}
            if isinstance(node, list):
                return [visit(value) for value in node]
            if isinstance(node, tuple):
                return tuple(visit(value) for value in node)
            return node

        value = visit(self.domain.space)
        return value, raw

    def __call__(self, new_id):
        _, raw = self.sample()
        return trial_doc(int(new_id), raw)

    def batch(self, new_ids):
        return [self(new_id) for new_id in new_ids]


def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    return AnnealingAlgo(domain, trials, seed, *args, **kwargs).batch(new_ids)


def suggest_batch(new_ids, domain, trials, seed, *args, **kwargs):
    return suggest(new_ids, domain, trials, seed, *args, **kwargs)
