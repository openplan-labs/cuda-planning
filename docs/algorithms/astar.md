# Batched A* and space-time A*

Two searches share one idea: on a unit-cost grid, the set of states at
cost *t* is exactly the *t*-th wavefront, so expanding a whole frontier
per sweep visits states in the same optimal order a serial A* or
Dijkstra would — with every cell of the frontier processed in parallel.
"A*" here names the problem semantics (optimal paths, Hart, Nilsson
and Raphael 1968), not a serial open list.

## `batched_astar`

Optimal single-agent paths for many `(start, goal)` queries at once:
one batched BFS from the goals (which doubles as a reusable heuristic
table), then an O(path length) gradient descent per query on the host.
Returns cost-optimal paths, `None` for unreachable queries.

## `space_time_astar`

The constrained low-level search of MAPF (Silver 2005, *Cooperative
pathfinding*, AIIDE): states are `(cell, time)` pairs, waiting is a
legal move, and a [`ReservationTable`](../api.md) supplies

- **vertex constraints** — cell `v` occupied at time `t`;
- **edge constraints** — encoded as `arrived_from[t, v]`: because at
  most one agent arrives anywhere per timestep, a single integer per
  `(t, cell)` rules out every swap.

Each timestep is one masked dilation of the reachable set — five
candidate predecessors per cell (4 moves + wait), checked against the
table — implemented as array shifts on the CPU and as one kernel
launch per timestep on CUDA, with the table resident on the device.

The settle rule matches pymapf: an agent may finish on its goal only
after the last vertex reservation touching it, so a returned path can
be extended by waiting forever.

### Bounded horizon

The search enumerates timesteps up to a `horizon` bound and is complete
only within it — a dense table over `(horizon, cells)` is the price of
making constraint checks O(1) array lookups for the whole frontier at
once. The default bound is generous for the instance sizes this
library targets and is a constructor parameter everywhere it matters.

::: cuplan.astar
