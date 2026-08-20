# Experiments — raw data

Raw measurements behind the docs
[Experiments section](https://openplan-labs.github.io/cuda-planning/experiments/methodology/),
written cell-by-cell by `python -m cuplan.benchmark.sweep` (RTX A2000
Laptop GPU 4 GB, driver 560.35.03 / CUDA 12.6, i7-11850H, Python
3.10.12, CuPy 14.2.0, NumPy 2.2.6, pymapf 0.8.0; 2026-08-20).

| File | Contents |
| :-- | :-- |
| `mapf.csv` | prioritized + PIBT, agents × grid × density × seed × solver |
| `bfs.csv` | batched distance-map builds, batch × grid × seed × backend |
| `vo.csv` | velocity obstacles, agents × seed × backend, 80 steps |
| `flocking.csv` | boids, agents × seed × backend, 200 steps |
| `phases.csv` | CUDA per-phase wall times, long format |
| `figures/` | light-mode charts (the docs carry light + dark) |

One row per measurement. `status` is `solved` / `unsolved` /
`timeout` / `skipped` / `error`; timeouts record the cap as the
runtime and are never extrapolated; skips record their reason in
`extra` (JSON). Medians and bands are computed at chart time from the
per-seed rows — nothing here is pre-aggregated.

## The MAPF CUDA arm is incomplete

The MAPF stage runs last and the CUDA driver failed partway through
it: **132 of `mapf.csv`'s 747 rows are `error` rows** carrying
`cudaErrorUnknown`. cuplan CUDA data exists for all of 32² and for
64² at 5% density only; 64² at 15%/25%, all of 128², and the whole
256² grid are absent for that solver. The 256² grid was never reached
by any solver. Every other CSV here is complete — the BFS, VO,
flocking and phase stages all finished before the fault. The
[methodology page](https://openplan-labs.github.io/cuda-planning/experiments/methodology/#what-the-sweep-did-not-cover)
carries the full accounting; the figures never plot an unmeasured
cell.

## Reproduce or extend

The sweep resumes from these CSVs, so a re-run *skips* the cells
already present — `error` rows included. Delete the rows you want
re-measured first:

```bash
grep -v ',error,' benchmarks/experiments/mapf.csv > mapf.tmp \
  && mv mapf.tmp benchmarks/experiments/mapf.csv
python -m cuplan.benchmark.sweep --stage all --out benchmarks/experiments
python -m cuplan.benchmark.figures
```
