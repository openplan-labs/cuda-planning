"""Kernel packaging and NVRTC compilation.

The compile test needs CuPy but not necessarily a device: NVRTC is a
pure compiler. It is skipped when CuPy is missing entirely.
"""

import pytest

from cuplan.kernels import kernel_names, kernel_source

EXPECTED = {"bfs", "flocking", "spacetime", "velocity_obstacles"}


def test_all_kernel_sources_ship():
    assert set(kernel_names()) == EXPECTED
    for name in EXPECTED:
        source = kernel_source(name)
        assert 'extern "C" __global__' in source


def test_kernels_compile_with_nvrtc():
    pytest.importorskip("cupy")
    from cupy.cuda.compiler import compile_using_nvrtc

    for name in EXPECTED:
        try:
            cubin = compile_using_nvrtc(
                kernel_source(name), options=("--std=c++11",)
            )
        except RuntimeError as error:  # pragma: no cover - env-dependent
            if "CUDA headers" in str(error):
                pytest.skip(
                    "CuPy lacks CUDA headers; pip install 'cupy-cuda12x[ctk]'"
                )
            raise
        assert cubin  # non-empty PTX/cubin
