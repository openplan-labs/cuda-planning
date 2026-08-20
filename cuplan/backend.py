"""Backend selection: NumPy reference or CUDA via CuPy.

``pip install cuda-planning`` alone gives the CPU reference backend;
installing the ``[cuda12]`` or ``[cuda11]`` extra adds the CUDA one.
CuPy needs only the NVIDIA driver at runtime — kernels are CUDA C
compiled on first use through NVRTC, so no CUDA toolkit install is
required on the host.
"""

from __future__ import annotations

import functools
import os
from typing import Literal

Backend = Literal["auto", "cpu", "cuda"]

_VALID = ("auto", "cpu", "cuda")


class CudaUnavailableError(RuntimeError):
    """Raised when ``backend="cuda"`` is requested but no device works."""


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return True if CuPy is importable and a CUDA device executes.

    The probe runs one tiny kernel rather than trusting the import: a
    machine with CuPy installed but no usable driver fails at launch
    time, and that is the failure this function must report.
    Set ``CUPLAN_FORCE_CPU=1`` to make it return False, which is how CI
    tests the fallback path on GPU machines.
    """
    if os.environ.get("CUPLAN_FORCE_CPU"):
        return False
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        # Exercise a real launch: allocation + elementwise kernel + copy.
        result = int((cupy.arange(4, dtype=cupy.int32) ** 2).sum())
        return result == 14
    except Exception:
        return False


def resolve_backend(backend: Backend = "auto") -> Literal["cpu", "cuda"]:
    """Map ``"auto"`` to the best available backend, validating the name.

    ``"auto"`` prefers CUDA when :func:`cuda_available` holds, otherwise
    falls back to the NumPy reference. ``"cuda"`` raises
    :class:`CudaUnavailableError` instead of silently degrading — a
    benchmark that quietly ran on the CPU is worse than one that failed.
    """
    if backend not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        if not cuda_available():
            raise CudaUnavailableError(
                "backend='cuda' requested but no working CUDA device was "
                "found. Install cuda-planning[cuda12] (or [cuda11]) and "
                "check `nvidia-smi`; use backend='auto' to fall back."
            )
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def get_cupy():
    """Import and return CuPy, raising a helpful error if absent."""
    try:
        import cupy
    except ImportError as error:  # pragma: no cover - exercised without cupy
        raise CudaUnavailableError(
            "CuPy is not installed. Install cuda-planning[cuda12] for "
            "CUDA 12.x drivers or cuda-planning[cuda11] for CUDA 11.x."
        ) from error
    return cupy
