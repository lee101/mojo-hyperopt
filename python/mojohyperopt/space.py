"""Recursive search-space traversal shared by the search algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np

from .hp import Distribution


def collect_distributions(space) -> dict[str, Distribution]:
    found: dict[str, Distribution] = {}
    seen: set[int] = set()

    def visit(node):
        if isinstance(node, Distribution):
            if id(node) in seen:
                return
            seen.add(id(node))
            previous = found.get(node.label)
            if previous is not None and previous is not node:
                raise ValueError(f"duplicate label {node.label!r}")
            found[node.label] = node
            for option in node.options:
                visit(option)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(space)
    return found


def evaluate(space, assignment: dict[str, Any]):
    if isinstance(space, Distribution):
        raw = assignment[space.label]
        if space.name in ("choice", "pchoice"):
            return evaluate(space.options[int(raw)], assignment)
        if space.cast_int:
            return int(raw)
        return raw
    if isinstance(space, dict):
        return {key: evaluate(value, assignment) for key, value in space.items()}
    if isinstance(space, list):
        return [evaluate(value, assignment) for value in space]
    if isinstance(space, tuple):
        return tuple(evaluate(value, assignment) for value in space)
    return space


def prior_distribution(node: Distribution, rng: np.random.Generator, size: int):
    name = node.name
    parameters = node.parameters
    if name == "uniform":
        return rng.uniform(parameters[0], parameters[1], size=size)
    if name == "quniform":
        draws = rng.uniform(parameters[0], parameters[1], size=size)
        return np.round(draws / parameters[2]) * parameters[2]
    if name == "loguniform":
        return np.exp(rng.uniform(parameters[0], parameters[1], size=size))
    if name == "qloguniform":
        draws = np.exp(rng.uniform(parameters[0], parameters[1], size=size))
        return np.round(draws / parameters[2]) * parameters[2]
    if name == "normal":
        return rng.normal(parameters[0], parameters[1], size=size)
    if name == "qnormal":
        draws = rng.normal(parameters[0], parameters[1], size=size)
        return np.round(draws / parameters[2]) * parameters[2]
    if name == "lognormal":
        return np.exp(rng.normal(parameters[0], parameters[1], size=size))
    if name == "qlognormal":
        draws = np.exp(rng.normal(parameters[0], parameters[1], size=size))
        return np.round(draws / parameters[2]) * parameters[2]
    if name == "randint":
        return rng.integers(int(parameters[0]), int(parameters[1]), size=size)
    if name == "choice":
        return rng.integers(0, len(node.options), size=size)
    if name == "pchoice":
        return rng.choice(
            len(node.options), size=size, p=np.asarray(node.probabilities)
        )
    raise NotImplementedError(f"unsupported distribution {name!r}")


def sample_prior(space, rng: np.random.Generator):
    raw: dict[str, Any] = {}

    def sample(node):
        if isinstance(node, Distribution):
            value = prior_distribution(node, rng, 1)[0]
            value = int(value) if node.name in ("randint", "choice", "pchoice") else float(value)
            raw[node.label] = value
            if node.name in ("choice", "pchoice"):
                return sample(node.options[int(value)])
            return int(value) if node.cast_int else value
        if isinstance(node, dict):
            return {key: sample(value) for key, value in node.items()}
        if isinstance(node, list):
            return [sample(value) for value in node]
        if isinstance(node, tuple):
            return tuple(sample(value) for value in node)
        return node

    value = sample(space)
    return value, raw
