# Install

## CPU only

```bash
pip install cuda-planning
```

NumPy is the only dependency. Everything works — same algorithms, same
results — through the vectorized reference backend. This is also what
runs in CI.

!!! note
    `cuda-planning` is the distribution name; the import name is
    `cuplan`. The package is prepared for PyPI but not yet published —
    until it is, install from source:
    `pip install git+https://github.com/openplan-labs/cuda-planning`.

## With CUDA

```bash
pip install 'cuda-planning[cuda12]'   # CUDA 12.x drivers
pip install 'cuda-planning[cuda11]'   # CUDA 11.x drivers
```

Requirements:

- An NVIDIA GPU and driver (`nvidia-smi` should work).
- **No CUDA toolkit install** and no root access: kernels are CUDA C
  compiled at runtime by NVRTC, and CuPy ships as pip wheels.

If CuPy raises *"Failed to find CUDA headers"* on first kernel launch
(recent CuPy versions resolve headers lazily), install the header
wheels too — still pip-only, still no sudo:

```bash
pip install 'cupy-cuda12x[ctk]'
```

Verify the device path end to end:

```python
import cuplan
print(cuplan.cuda_available())   # True when a kernel actually ran
```

`backend="auto"` falls back to the CPU silently; `backend="cuda"`
raises instead — a benchmark that quietly ran on the CPU is worse than
one that failed. Setting `CUPLAN_FORCE_CPU=1` disables the device probe,
which is how the fallback path is tested on GPU machines.

## Development install

```bash
git clone https://github.com/openplan-labs/cuda-planning.git
cd cuda-planning
pip install -e '.[dev]'       # + '.[cuda12]' on a GPU machine
ruff check . && pytest
```

CUDA equivalence tests skip automatically without a device; see
[CONTRIBUTING.md](https://github.com/openplan-labs/cuda-planning/blob/main/CONTRIBUTING.md)
for the GPU-less workflow.
