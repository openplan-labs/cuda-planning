# Methodology

Every figure in this section was produced by one sweep of
[`cuplan.benchmark.sweep`](https://github.com/openplan-labs/cuda-planning/blob/main/cuplan/benchmark/sweep.py)
on the machine below, on 2026-08-20, and every chart states its own
conditions in its caption. A number without conditions is not a
result; this page is the conditions.

## Hardware and software

| Component | Value |
| :-- | :-- |
| GPU | NVIDIA RTX A2000 Laptop GPU, 4 GB VRAM (3.7 GB usable) |
| Driver | 560.35.03, CUDA 12.6; kernels compiled at runtime via NVRTC |
| CPU | Intel Core i7-11850H (8 cores / 16 threads, 2.50 GHz base) |
| RAM | 32 GB |
| Python | 3.10.12 (Linux x86_64) |
| cuplan | commit under test, editable install |
| CuPy | 14.2.0 (`cupy-cuda12x`) |
| NumPy | 2.2.6 (the CPU backend) |
| pymapf | 0.8.0 |
| Matplotlib | 3.10.9, styled by `frontier.mplstyle` |

The CPU backend is single-process NumPy: vectorized, but one core's
worth of orchestration. The comparison is "this library's GPU path
vs this library's reference path vs pymapf as published", not a
tuned-CPU-vs-tuned-GPU shootout.

## Instance generation

MAPF instances come from
[`cuplan.benchmark.scenarios.random_scenario`](https://github.com/openplan-labs/cuda-planning/blob/main/cuplan/benchmark/scenarios.py):
obstacles sampled i.i.d. at the stated density, starts and goals drawn
without replacement from one connected component (verified by flood
fill), so every goal is reachable. The seed fixes the instance —
`random_scenario(64, 128, 0.15, seed=1)` is the same problem on every
machine — and the *same instance object* is handed to every solver.
Velocity-obstacle runs start agents on a circle with antipodal goals
(the all-cross stress case); flocking starts agents uniformly in a
square scaled to hold density constant.

## Axes

The axes below are what the sweep was *configured* to cover. The MAPF
stage did not finish; [what the sweep did not
cover](#what-the-sweep-did-not-cover) says exactly which cells are
absent and why, and no figure in this section draws a cell that was
not measured.

| Axis | Configured values | Reached |
| :-- | :-- | :-- |
| Grids (MAPF) | 32², 64², 128², 256² | 32², 64², 128² |
| Obstacle density | 5%, 15%, 25% | all three on 32² and 64²; 5% only on 128² |
| Agents (MAPF) | 8, 16, 32, 64, 128, 256, 512 | all, where the instance fits |
| Solvers | pymapf, cuplan CPU, cuplan CUDA | all; CUDA partial (see below) |
| BFS batches | 16, 64, 256, 1024, 2048 sources on 64²–256² | complete |
| VO agents | 8–1024 (80 steps) | complete |
| Flocking agents | 64–8192 (200 steps) | complete |
| Phase cells | BFS 16–1024 on 256², VO 16–1024, flocking 256–4096, prioritized (64²,64), (64²,256), (128²,256) | complete |
| Seeds | 0, 1, 2 — every plotted line is the median, bands are min–max | complete |

Cells where the agents would not fit are never generated, which caps
the 32² grid at 128 agents: the sweep refuses an instance whose
starts and goals would occupy more than half the free cells.

## Timing protocol

Wall time for the full `solve()` (or `run()`) call, **host/device
transfers included** — the number a user sees. CUDA runs are preceded
by one small warm-up call so NVRTC compilation and context creation
are not billed to the measurement (first-call compile cost is real
but paid once per machine — CuPy caches compiled kernels on disk —
so it does not belong inside a scaling curve).
Per-phase numbers on the [phases page](phases.md) come from separate
profiled runs with device synchronization at phase boundaries;
profiling adds a small cost, so headline totals always come from the
unprofiled runs.

## What "success" means

A run is **solved** only if the returned joint plan passes cuplan's
independent validator (no vertex conflicts, no edge swaps, implicit
goal-parking included) — solver self-reporting is not trusted.
**Unsolved** means the solver terminated and reported failure;
prioritized planning and PIBT are incomplete, so unsolved cells on
crowded instances are results, not errors. **Timeout** means the
solver was stopped at the cap — 60 s per instance for pymapf, 300 s
for cuplan — and is plotted as a hollow triangle at the cap, never
extrapolated. **Skipped** cells were not run, for the reasons below.

## What the sweep did not cover

An honest sweep on a 4 GB laptop GPU has edges. Some were planned;
one was not.

### The unplanned one: the MAPF CUDA arm is partial

The stages run in the order BFS → velocity obstacles → flocking →
phases → MAPF, and **the CUDA driver failed partway through the last
stage.** Every context creation after that point returned
`cudaErrorUnknown`, and the fault outlived the sweep process — a
stale `nvidia_uvm` state that needs a module reload to clear. The
sweep did what it is built to do: it recorded each affected cell as
an `error` row carrying the driver's message, and kept going, so the
CPU and pymapf arms of every remaining cell are complete and real.

The result, in `mapf.csv`:

| | measured | absent |
| :-- | :-- | :-- |
| cuplan CUDA, 32² | all 3 densities, 8–128 agents | — |
| cuplan CUDA, 64² | 5% density, 8–128 agents (the 128-agent cells rest on 1–2 seeds) | 15% and 25% at every agent count; 5% at 256 and 512 |
| cuplan CUDA, 128² | — | everything |
| any solver, 256² | — | everything |

132 of the 747 `mapf.csv` rows are `error` rows. Figures never plot
them: a panel whose CUDA rows are all errors is labelled
"CUDA arm lost — device fault" rather than left to read as
"not measured yet", and success-rate cells with nothing to report
show an em dash. The two MAPF pages and the
[summary](summary.md#what-is-missing) say per-figure which lines are
affected.

What survives unaffected: everything in the BFS, velocity-obstacle,
flocking and phase stages, all of which completed before the fault on
their full configured axes; and the entire cuplan-CPU-versus-pymapf
comparison, which spans all three grids and all three densities.

Because the checkpoint file keys on the cell, re-running the sweep as
committed would *skip* those rows rather than retry them. Delete the
`error` rows first:

```bash
grep -v ',error,' benchmarks/experiments/mapf.csv > mapf.tmp \
  && mv mapf.tmp benchmarks/experiments/mapf.csv
python -m cuplan.benchmark.sweep --stage mapf --out benchmarks/experiments
```

### The planned ones

- **pymapf above 128² grids.** Its per-agent Dijkstra tables already
  need minutes per instance at 128²; every larger cell would be a
  timeout row that costs an hour to write. Recorded as skipped.
- **pymapf timeout escalation.** Once every seed of a cell timed out,
  larger agent counts on the same (family, grid, density) axis are
  recorded as skipped instead of burning 60 s × 3 seeds each.
- **CPU backend caps**: batched BFS above 1024 sources (a 2048-map
  build costs ~2 minutes per seed), velocity obstacles above 1024
  agents and flocking above 2048 agents (the O(n²) NumPy temporaries
  outgrow host RAM). The CUDA side runs the full axis.
- **pymapf is absent from velocity obstacles and flocking** by
  design, not omission: its simulator updates agents *sequentially
  within* a timestep, so a wall-clock comparison would time two
  different problems — see the
  [semantics note](../algorithms/velocity-obstacles.md).
- **CBS/conflict-based search is not benchmarked** because cuplan does
  not implement it yet (see the [roadmap](../roadmap.md)); the
  comparison covers the algorithms both libraries share.

Raw CSVs (one row per measurement, statuses included) and the exact
commands are committed under
[`benchmarks/experiments/`](https://github.com/openplan-labs/cuda-planning/tree/main/benchmarks/experiments).
Reproduce with:

```bash
pip install 'cuda-planning[cuda12,benchmark]'
python -m cuplan.benchmark.sweep --stage all --out benchmarks/experiments
python -m cuplan.benchmark.figures
```

The sweep checkpoints every cell to CSV as it finishes; a killed run
resumes where it stopped.
