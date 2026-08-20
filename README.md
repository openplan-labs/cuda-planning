<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/openplan-labs/branding/main/assets/logo/mark-dark.svg">
  <img src="https://raw.githubusercontent.com/openplan-labs/branding/main/assets/logo/mark-accent.svg" width="44" alt="OpenPlan Labs">
</picture>

# cuda-planning

[![CI](https://github.com/openplan-labs/cuda-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/openplan-labs/cuda-planning/actions/workflows/ci.yml)
[![docs](https://github.com/openplan-labs/cuda-planning/actions/workflows/docs.yml/badge.svg)](https://openplan-labs.github.io/cuda-planning/)
[![PyPI](https://img.shields.io/badge/PyPI-not%20yet%20published-6d8298)](https://openplan-labs.github.io/cuda-planning/install/)
[![License: MIT](https://img.shields.io/badge/license-MIT-6d8298)](LICENSE)

CUDA-accelerated multi-agent path finding (MAPF) and planning — the GPU
sibling of [pymapf](https://github.com/openplan-labs/pymapf). Every
algorithm has a vectorized NumPy reference implementation and a CUDA
implementation behind one `backend="auto" | "cpu" | "cuda"` API, with
CPU == CUDA equivalence enforced by tests. Kernels are CUDA C compiled
at runtime through CuPy's NVRTC bindings, so a machine needs an NVIDIA
driver but no CUDA toolkit — and without a GPU at all, `pip install
cuda-planning` still gives the full library on the reference backend.
The GPU wins by batching (one distance map per agent, one wavefront per
timestep, one thread per velocity sample), not by forcing serial
searches onto device threads; each algorithm's docs say exactly which
part runs where. Solvers included today are fast and *incomplete* —
optimal search (CBS and friends) is on the roadmap, not in the box.

## Install

```bash
pip install cuda-planning            # CPU reference backend (NumPy only)
pip install 'cuda-planning[cuda12]'  # + CUDA, for CUDA 12.x drivers
pip install 'cuda-planning[cuda11]'  # + CUDA, for CUDA 11.x drivers
```

Not yet on PyPI — until then:
`pip install git+https://github.com/openplan-labs/cuda-planning`.
If CuPy reports missing CUDA headers on first launch, add them with
`pip install 'cupy-cuda12x[ctk]'` (pip wheels; no toolkit, no sudo).
Details in the [install guide](https://openplan-labs.github.io/cuda-planning/install/).

## Quickstart

```python
import numpy as np
from cuplan import Grid, Problem, Agent, PIBT, distance_maps

# A 64x64 grid with random obstacles; truthy = blocked.
rng = np.random.default_rng(0)
obstacles = rng.random((64, 64)) < 0.15
obstacles[0, 0] = obstacles[60, 60] = False
grid = Grid(obstacles)

# One exact distance map per agent, in a single GPU batch.
tables = distance_maps(grid, sources=[(0, 0), (60, 60)], backend="auto")

# Solve a MAPF instance; backend="auto" uses the GPU when one works.
problem = Problem(grid, [Agent("a", (0, 0), (60, 60)),
                         Agent("b", (60, 60), (0, 0))])
solution = PIBT(backend="auto").solve(problem)
print(solution.sum_of_costs, solution.makespan, solution.is_valid())
```

Paths, costs, and conflict rules match pymapf exactly, so scenarios and
results transfer between the two libraries unchanged.

## Algorithms

| Algorithm | Reference | Status |
| :-- | :-- | :-- |
| [Batched BFS distance maps](https://openplan-labs.github.io/cuda-planning/algorithms/bfs/) | frontier-parallel BFS (Merrill et al. 2012) | ✓ CPU + CUDA |
| [Batched A* shortest paths](https://openplan-labs.github.io/cuda-planning/algorithms/astar/) | Hart, Nilsson & Raphael 1968 | ✓ CPU + CUDA |
| [Space-time A* + reservations](https://openplan-labs.github.io/cuda-planning/algorithms/astar/) | Silver 2005 | ✓ CPU + CUDA |
| [Prioritized planning](https://openplan-labs.github.io/cuda-planning/algorithms/prioritized/) | Erdmann & Lozano-Pérez 1987 | ✓ CPU + CUDA |
| [PIBT](https://openplan-labs.github.io/cuda-planning/algorithms/pibt/) | Okumura et al. 2022 | ✓ CPU + CUDA |
| [Velocity obstacles](https://openplan-labs.github.io/cuda-planning/algorithms/velocity-obstacles/) | Fiorini & Shiller 1998 | ✓ CPU + CUDA |
| [Flocking (Boids)](https://openplan-labs.github.io/cuda-planning/algorithms/flocking/) | Reynolds 1987 | ✓ CPU + CUDA |
| CBS · LaCAM · LNS · SIPP · NMPC | [roadmap](https://openplan-labs.github.io/cuda-planning/roadmap/) | planned |

## Benchmarks

![Prioritized planning wall time vs agents, 64x64 grid: cuplan-cuda fastest, cuplan-cpu close, pymapf orders of magnitude slower](benchmarks/results/prioritized-scaling.png)

Measured on an NVIDIA RTX A2000 Laptop GPU (4 GB, driver 560.35) and
its host CPU; random grids at 15% obstacle density, identical seeded
instances for every solver, medians over 3 seeds; wall time includes
host/device transfers. Same-cost check: cuplan's CPU and CUDA backends
return identical solutions.

| Workload | pymapf | cuplan CPU | cuplan CUDA |
| :-- | --: | --: | --: |
| Prioritized planning, 64×64, 128 agents | 46.34 s | 0.64 s | **0.29 s** (159×) |
| PIBT, 64×64, 128 agents | 1.26 s | 0.12 s | **0.05 s** (24×) |
| Batched BFS, 512 maps on 256×256 | — | 26.37 s | **1.03 s** (26×) |
| Velocity obstacles, 128 agents × 80 steps | — | 1.37 s | **0.44 s** (3.1×) |
| Prioritized planning, 128×128, 256 agents | — | 4.53 s | **1.14 s** (4.0×) |
| Batched BFS, 1024 maps on 256×256 | — | 54.25 s | **1.86 s** (29×) |

The wins come from batch size, and small instances go the other way:
at 16 agents the velocity-obstacle step is *faster on the CPU*
(0.028 s vs 0.043 s — launch overhead dominates), and prioritized
planning on 32×32 grids gains little. The crossover sits around a few
dozen agents; below it, use `backend="cpu"`.

Full tables, conditions, and charts: [`benchmarks/results/`](benchmarks/results/)
and the [benchmark docs](https://openplan-labs.github.io/cuda-planning/benchmarks/).
Reproduce with `python -m cuplan.benchmark` (add
`pip install 'cuda-planning[benchmark]'` for the pymapf baselines and
charts).

## Documentation

Guides, per-algorithm semantics and parallelization notes, and the API
reference live at
**[openplan-labs.github.io/cuda-planning](https://openplan-labs.github.io/cuda-planning/)**.

## Contributing

Contributions are welcome, with or without a GPU — the CPU reference
backend carries the full test suite, and CUDA equivalence tests skip
cleanly on machines without a device. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and the
kernel/reference contract, and the
[roadmap](https://openplan-labs.github.io/cuda-planning/roadmap/) for
where help is most useful.

## License

[MIT](LICENSE) © Erwin Lejeune. Part of
[OpenPlan Labs](https://github.com/openplan-labs).
