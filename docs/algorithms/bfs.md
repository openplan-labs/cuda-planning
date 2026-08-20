# Batched BFS distance maps

`cuplan.distance_maps(grid, sources, backend="auto")` computes the
exact 4-connected distance from each of *n* sources to every cell —
*n* flood fills as one batch.

## Why it is the workhorse

Every serious MAPF implementation runs on exact goal-distance tables:
they are the only admissible heuristic worth having on a map with
walls, and PIBT queries them millions of times. pymapf builds them with
one backward Dijkstra per agent, serially. On a unit-cost grid Dijkstra
*is* breadth-first search, and BFS is a wavefront — every frontier cell
can be expanded simultaneously.

## Parallelization

The batch is a `(n, height, width)` volume advanced one wave per sweep:

- **CPU reference** — four array shifts OR-ed together per wave,
  vectorized over the whole batch with NumPy.
- **CUDA** — a gather ("pull") kernel: at wave *t* every unlabelled
  free cell checks whether any neighbour was labelled *t−1*
  (Merrill, Garland and Grimshaw 2012, *Scalable GPU graph traversal*,
  PPoPP). Pull needs no atomics and stays coalesced; the cost of
  touching settled cells each wave is amortized across the batch.

The number of kernel launches equals the graph diameter, not the
number of sources — batching 512 sources costs barely more than 16.
Measured scaling is in [Benchmarks](../benchmarks.md).

## Semantics

- Returns `int32` distances, `-1` for unreachable cells (blocked cells
  included).
- Matches `pymapf.algorithms.search.distance_table` values exactly
  (tested against a reference Python BFS, and CPU == CUDA is asserted
  on GPU machines).

::: cuplan.bfs
