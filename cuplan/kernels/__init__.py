"""CUDA C kernel sources and their NVRTC loader.

Kernels ship as ``.cu`` source files inside the wheel and are compiled
on first use through CuPy's :class:`cupy.RawModule` (NVRTC). CuPy caches
compiled cubins on disk, so the cost is paid once per machine, not once
per process.
"""

from __future__ import annotations

import functools
from importlib import resources

from ..backend import get_cupy

_OPTIONS = ("--std=c++11",)


def kernel_source(name: str) -> str:
    """Return the CUDA C source of ``kernels/<name>.cu``."""
    return (resources.files(__package__) / f"{name}.cu").read_text()


def kernel_names() -> list[str]:
    """List the kernel source files bundled with the package."""
    return sorted(
        path.name[:-3]
        for path in resources.files(__package__).iterdir()
        if path.name.endswith(".cu")
    )


@functools.cache
def load_module(name: str):
    """Compile ``kernels/<name>.cu`` and return the ``cupy.RawModule``."""
    cupy = get_cupy()
    return cupy.RawModule(code=kernel_source(name), options=_OPTIONS)


def get_kernel(module_name: str, kernel: str):
    """Return a launchable ``cupy.RawKernel`` from a bundled module."""
    return load_module(module_name).get_function(kernel)
