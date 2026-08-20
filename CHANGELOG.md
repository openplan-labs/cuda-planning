# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-20

Initial release.

### Added

- `cuplan.Grid`, `Problem`, `Solution` — problem vocabulary matching
  pymapf's semantics (4-connected grids, unit costs, vertex/edge
  conflicts, sum-of-costs, makespan).
- Dual-backend execution: NumPy reference (`backend="cpu"`) and CUDA C
  kernels compiled at runtime through CuPy/NVRTC (`backend="cuda"`),
  selected per call with `backend="auto"`.
- `distance_maps` — batched grid BFS / flood fill (one distance map per
  source, frontier-parallel on the GPU).
- `batched_astar` — optimal shortest paths for batches of
  (start, goal) queries.
- `space_time_astar` + `ReservationTable` — constrained space-time
  search with vertex and edge (swap) reservations.
- `PrioritizedPlanning` — cooperative A* with a device-resident
  reservation table.
- `PIBT` — Priority Inheritance with Backtracking with a batched
  distance oracle and vectorized candidate evaluation.
- `VelocityObstacleSim` — decentralized velocity-obstacle avoidance,
  one thread per (agent, sample).
- `FlockingSim` — Reynolds Boids with GPU force accumulation.
- `cuplan.benchmark` — harness running identical scenarios through
  pymapf and both cuplan backends, with CSV/Markdown/chart output.
- Roadmap stubs with parallelization notes: CBS, LaCAM, LNS, SIPP,
  NMPC.

[0.1.0]: https://github.com/openplan-labs/cuda-planning/releases/tag/v0.1.0
