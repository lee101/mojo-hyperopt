"""ctypes bridge to the single Mojo compilation unit."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "hyperopt.mojo")
BUILD = os.path.join(ROOT, "build", "build.sh")
LIB = os.environ.get("MOJO_HYPEROPT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-hyperopt.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mho_gmm1_lpdf": ([I, I, I, I, I, I, F, F, I, I, F, I, I], I),
    "mho_lgmm1_lpdf": ([I, I, I, I, I, I, F, F, I, I, F, I, I], I),
    "mho_gmm1_lpdf_gpu": ([I, I, I, I, I, I, I, I], I),
    "mho_adaptive_parzen_normal": (
        [I, I, F, F, F, I, I, I, I, I],
        I,
    ),
    "mho_categorical_lpdf": ([I, I, I, I, I], I),
    "mho_anneal_bounds": ([I, I, F, F, F, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_HYPEROPT_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_HYPEROPT_LIB does not exist: {LIB}")
    stale = (
        force
        or not os.path.exists(LIB)
        or os.path.getmtime(LIB) < os.path.getmtime(SRC)
        or os.path.getmtime(LIB) < os.path.getmtime(BUILD)
    )
    if stale:
        proc = subprocess.run(
            ["bash", BUILD], cwd=ROOT, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0:
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    if not os.path.exists(LIB):
        raise BuildError(f"build did not create {LIB}")
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    return int(array.ctypes.data)


def checked_call(name: str, *args) -> None:
    """Call a status-returning kernel and surface rejected ABI arguments."""
    if not getattr(lib(), name)(*args):
        raise RuntimeError(f"Mojo kernel {name} rejected its arguments")


if __name__ == "__main__":
    print(build(force=True))
