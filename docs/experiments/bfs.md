# Batched BFS

The primitive underneath everything else — one exact distance map per
source, batched into a single `(N, H, W)` wavefront — measured alone.
Conditions for the whole page: random grids at 15% obstacle density,
sources drawn from the free space with a fixed seed, seeds {0, 1, 2},
medians with min–max bands, wall time including transfers. CPU runs
stop at 1024 sources (a 2048-map build costs about two minutes per
seed on this CPU); the CUDA axis continues to 2048.

## Build time

![Batched BFS build time vs batch size, one panel per grid size](../assets/experiments/bfs-scaling.png#only-light)
![Batched BFS build time vs batch size, one panel per grid size](../assets/experiments/bfs-scaling-dark.png#only-dark)

*Median build time for a batch of distance maps, log–log. Grids 64²,
128², 256² at 15% density; seeds {0, 1, 2}; no timeouts occurred.*

Three readings:

- **There is no crossover in this family.** Even 16 maps on a 64²
  grid build faster on the GPU (3.3 ms vs 9.3 ms). The workload is
  born batched — the wavefront advances every map with every sweep —
  so the kernel is never launch-bound the way the small
  velocity-obstacle steps are.
- **The gap widens with both axes.** 17× at (64², 1024 maps), 23× at
  (128², 1024), 33× at (256², 1024): 62.1 s of NumPy sweeps against
  1.87 s on the device. Kernel launches scale with graph *diameter*,
  not batch size, so bigger batches amortize the same number of
  launches over more parallel work.
- **The CPU line bends up; the CUDA line does not.** CPU throughput
  *falls* as batches grow (1 720 → 1 392 maps/s on 64²) because the
  `(N, H, W)` boolean temporaries outgrow cache, while CUDA
  throughput *rises* (4 860 → 25 800 maps/s) as the batch fills the
  device — the saturation story on the
  [phases and crossover page](phases.md).

## Selected medians

| grid | maps | cuplan CPU | cuplan CUDA | speedup |
| :-- | --: | --: | --: | --: |
| 64² | 1024 | 0.736 s | 0.044 s | 17× |
| 128² | 1024 | 7.00 s | 0.304 s | 23× |
| 256² | 256 | 11.15 s | 0.593 s | 19× |
| 256² | 1024 | 62.09 s | **1.87 s** | 33× |
| 256² | 2048 | — (capped) | 3.62 s | — |

*Same conditions as the figure. The 2048-map CPU cell was cut for
time, not ability — see [methodology](methodology.md).*

Why this page matters beyond itself: the MAPF solvers consume these
tables. pymapf builds the equivalent oracle with one serial Dijkstra
per agent, which is most of the gap on the
[prioritized planning](prioritized.md) and [PIBT](pibt.md) pages.
