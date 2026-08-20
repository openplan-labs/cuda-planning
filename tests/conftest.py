"""Shared fixtures: parametrize over available backends."""

from __future__ import annotations

import pytest

from cuplan.backend import cuda_available


def _backends():
    yield "cpu"
    if cuda_available():
        yield "cuda"


@pytest.fixture(params=list(_backends()))
def backend(request):
    """Every available backend; CUDA only when a device works."""
    return request.param


needs_cuda = pytest.mark.skipif(
    not cuda_available(), reason="no working CUDA device"
)
