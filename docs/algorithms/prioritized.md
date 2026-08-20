# Prioritized planning

`cuplan.PrioritizedPlanning` — cooperative A* (Erdmann and
Lozano-Pérez 1987; Silver 2005): agents plan one at a time in priority
order, each treating the already-planned agents as moving obstacles via
space-time reservations.

## Guarantees, stated plainly

Prioritized planning is **incomplete and suboptimal**: a bad priority
order can fail on solvable instances (Ma et al. 2019, *Searching with
consistent prioritization for MAPF*, AAAI), and the bounded search
horizon is a second, independent source of incompleteness. Each agent's
own path *is* time-optimal given the reservations it faces. Returned
solutions are always conflict-free — validity is checked, not assumed.

## Parallelization

The priority loop is the algorithm, so it stays sequential on the
host. Everything inside one iteration moves to the device:

- the per-agent constrained search runs as a
  [space-time wavefront](astar.md) — one kernel launch per timestep,
  all frontier cells in parallel;
- the reservation table lives on the device across the whole solve, so
  planning agent *k* never copies the *k−1* previous paths back and
  forth;
- the up-front solvability check and horizon bound come from one
  [batched BFS](bfs.md) over all goals.

This is the honest shape of GPU prioritized planning: the sequential
skeleton is unchanged, and the O(cells) work per timestep inside it is
what parallelizes.

::: cuplan.prioritized
