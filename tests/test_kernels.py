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
    from cupy_backends.cuda.libs import nvrtc

    # NVRTC is called directly (not through cupy.cuda.compiler, which
    # queries the device for its arch) with an explicit compute_70, so
    # this compiles on driverless machines — it is what CI runs.
    try:
        nvrtc.getVersion()
    except Exception:  # pragma: no cover - env-dependent
        pytest.skip("libnvrtc unavailable; pip install 'cupy-cuda12x[ctk]'")
    for name in EXPECTED:
        program = nvrtc.createProgram(
            kernel_source(name), f"{name}.cu", [], []
        )
        try:
            nvrtc.compileProgram(
                program, ["--std=c++11", "-arch=compute_70"]
            )
        except Exception as error:
            log = nvrtc.getProgramLog(program)
            raise AssertionError(
                f"kernel {name!r} failed to compile:\n{log}"
            ) from error
        assert nvrtc.getPTX(program)  # non-empty PTX
