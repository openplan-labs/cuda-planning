import pytest

from cuplan.backend import (
    CudaUnavailableError,
    cuda_available,
    resolve_backend,
)


def test_cpu_always_resolves():
    assert resolve_backend("cpu") == "cpu"


def test_auto_resolves_to_something():
    assert resolve_backend("auto") in ("cpu", "cuda")


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        resolve_backend("tpu")


def test_cuda_matches_probe():
    if cuda_available():
        assert resolve_backend("cuda") == "cuda"
    else:
        with pytest.raises(CudaUnavailableError):
            resolve_backend("cuda")


def test_force_cpu_env(monkeypatch):
    monkeypatch.setenv("CUPLAN_FORCE_CPU", "1")
    cuda_available.cache_clear()
    try:
        assert cuda_available() is False
        assert resolve_backend("auto") == "cpu"
    finally:
        monkeypatch.delenv("CUPLAN_FORCE_CPU")
        cuda_available.cache_clear()
