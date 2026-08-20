# cuda-planning

CUDA-accelerated multi-agent path finding (MAPF) and planning — the GPU
sibling of [pymapf](https://github.com/openplan-labs/pymapf). Every
algorithm ships a vectorized NumPy reference implementation and a CUDA
realization behind one API:

```python
from cuplan import Grid, Problem, Agent, PrioritizedPlanning

grid = Grid.empty(64, 64)
problem = Problem(grid, [Agent("a", (0, 0), (63, 63)),
                         Agent("b", (63, 0), (0, 63))])
solution = PrioritizedPlanning(backend="auto").solve(problem)
print(solution.sum_of_costs, solution.is_valid())
```

`backend="auto"` uses CUDA when a device works and the NumPy reference
otherwise; `"cpu"` and `"cuda"` pin the choice. The CUDA kernels are
plain CUDA C sources compiled at runtime through CuPy's NVRTC bindings,
so installation needs an NVIDIA driver but **no CUDA toolkit**.

## What it does — and does not do

The GPU wins by *batching*: one distance map per agent, one wavefront
sweep per timestep, one thread per (agent, velocity-sample) pair.
Inherently sequential steps — priority orders, PIBT's inheritance
chains — stay on the host, and each algorithm page says exactly which
part runs where. Optimal multi-agent solvers (CBS and friends) are
[on the roadmap](roadmap.md), not in the box: the two solvers shipped
today, prioritized planning and PIBT, are fast and *incomplete*.

Problem semantics match pymapf exactly — 4-connected grids, unit edge
costs, vertex and edge conflicts, sum-of-costs and makespan — so
scenarios and results transfer between the libraries unchanged, which
is what makes the [benchmarks](benchmarks.md) apples-to-apples.

## Algorithms

| Algorithm | Status | Backends |
| :-- | :-- | :-- |
| [Batched BFS distance maps](algorithms/bfs.md) | implemented | CPU, CUDA |
| [Batched A* (shortest paths)](algorithms/astar.md) | implemented | CPU, CUDA |
| [Space-time A* + reservations](algorithms/astar.md) | implemented | CPU, CUDA |
| [Prioritized planning](algorithms/prioritized.md) | implemented | CPU, CUDA |
| [PIBT](algorithms/pibt.md) | implemented | CPU, CUDA |
| [Velocity obstacles](algorithms/velocity-obstacles.md) | implemented | CPU, CUDA |
| [Flocking (Boids)](algorithms/flocking.md) | implemented | CPU, CUDA |
| CBS, LaCAM, LNS, SIPP, NMPC | [planned](roadmap.md) | — |

## Where to start

- [Install](install.md) — with or without a GPU.
- [Benchmarks](benchmarks.md) — measured numbers, conditions included.
- [API reference](api.md) — every public symbol.
