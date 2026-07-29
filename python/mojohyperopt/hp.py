"""Hyperopt-compatible constructors for the supported search-space nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Distribution:
    label: str
    name: str
    parameters: tuple[float | int, ...] = ()
    options: tuple[Any, ...] = ()
    probabilities: tuple[float, ...] = ()
    cast_int: bool = False


def _label(label) -> str:
    if not isinstance(label, str):
        raise TypeError("require string label")
    return label


def _parameters(args, kwargs, names) -> tuple[float, ...]:
    if len(args) > len(names):
        raise TypeError(f"expected at most {len(names)} distribution arguments")
    values = dict(zip(names, args))
    for key, value in kwargs.items():
        if key not in names:
            raise TypeError(f"unexpected keyword argument {key!r}")
        if key in values:
            raise TypeError(f"multiple values for argument {key!r}")
        values[key] = value
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"missing distribution argument {missing[0]!r}")
    return tuple(float(values[name]) for name in names)


def choice(label, options):
    options = tuple(options)
    if not options:
        raise ValueError("hp.choice requires at least one option")
    return Distribution(_label(label), "choice", options=options)


def pchoice(label, p_options):
    pairs = tuple(p_options)
    if not pairs:
        raise ValueError("hp.pchoice requires at least one option")
    probabilities = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")
    return Distribution(
        _label(label),
        "pchoice",
        options=tuple(pair[1] for pair in pairs),
        probabilities=tuple(float(p) for p in probabilities),
    )


def randint(label, *args, **kwargs):
    if kwargs:
        if args:
            raise TypeError("randint accepts positional bounds or keyword bounds")
        if "high" in kwargs and "low" in kwargs:
            args = (kwargs["low"], kwargs["high"])
        elif "low" in kwargs and "high" not in kwargs:
            args = (kwargs["low"],)
        else:
            raise TypeError("randint requires low or low and high")
    if len(args) == 1:
        low, high = 0, int(args[0])
    elif len(args) == 2:
        low, high = int(args[0]), int(args[1])
    else:
        raise TypeError("randint expects high, or low and high")
    if high <= low:
        raise ValueError("high must exceed low")
    return Distribution(_label(label), "randint", (low, high))


def uniform(label, *args, **kwargs):
    low, high = _parameters(args, kwargs, ("low", "high"))
    if high <= low:
        raise ValueError("low should be less than high")
    return Distribution(_label(label), "uniform", (low, high))


def uniformint(label, *args, **kwargs):
    low, high = _parameters(args, kwargs, ("low", "high"))
    if high <= low:
        raise ValueError("low should be less than high")
    return Distribution(_label(label), "quniform", (low, high, 1.0), cast_int=True)


def quniform(label, *args, **kwargs):
    low, high, q = _parameters(args, kwargs, ("low", "high", "q"))
    if high <= low or q <= 0:
        raise ValueError("require low < high and q > 0")
    return Distribution(_label(label), "quniform", (low, high, q))


def loguniform(label, *args, **kwargs):
    low, high = _parameters(args, kwargs, ("low", "high"))
    if high <= low:
        raise ValueError("low should be less than high")
    return Distribution(_label(label), "loguniform", (low, high))


def qloguniform(label, *args, **kwargs):
    low, high, q = _parameters(args, kwargs, ("low", "high", "q"))
    if high <= low or q <= 0:
        raise ValueError("require low < high and q > 0")
    return Distribution(_label(label), "qloguniform", (low, high, q))


def normal(label, *args, **kwargs):
    mu, sigma = _parameters(args, kwargs, ("mu", "sigma"))
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return Distribution(_label(label), "normal", (mu, sigma))


def qnormal(label, *args, **kwargs):
    mu, sigma, q = _parameters(args, kwargs, ("mu", "sigma", "q"))
    if sigma <= 0 or q <= 0:
        raise ValueError("sigma and q must be positive")
    return Distribution(_label(label), "qnormal", (mu, sigma, q))


def lognormal(label, *args, **kwargs):
    mu, sigma = _parameters(args, kwargs, ("mu", "sigma"))
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return Distribution(_label(label), "lognormal", (mu, sigma))


def qlognormal(label, *args, **kwargs):
    mu, sigma, q = _parameters(args, kwargs, ("mu", "sigma", "q"))
    if sigma <= 0 or q <= 0:
        raise ValueError("sigma and q must be positive")
    return Distribution(_label(label), "qlognormal", (mu, sigma, q))
