"""Batched shortest paths and constrained space-time search.

Two searches live here, both realized as frontier-parallel wavefronts:

* :func:`batched_astar` — optimal single-agent paths for many
  ``(start, goal)`` queries at once. On a 4-connected grid with unit
  edge costs, A* (Hart, Nilsson and Raphael 1968) and Dijkstra return
  the same paths; the wavefront expands exactly the cost-``t`` band per
  sweep, so the batched flood fill *is* the optimal search, with the
  entire batch advanced by every sweep. Paths are then extracted by
  gradient descent on the distance maps — O(path length) host work.

* :func:`space_time_astar` — the constrained low-level search of MAPF
  (Silver 2005): states are ``(cell, t)`` pairs, wait moves are allowed,
  and vertex/edge reservations from a :class:`~cuplan.reservations.ReservationTable`
  are honoured. Since every edge costs one timestep, the set of states
  reachable at time ``t`` is exactly the cost-``t`` band, so one masked
  dilation per timestep enumerates the search space in optimal order —
  every cell in parallel on the CUDA backend.

Both return cost-optimal paths (given the horizon bound, for the
constrained search). "A*" names the problem semantics, not a serial
open list: with unit costs the wavefront explores in the same optimal
order without one.
"""

from __future__ import annotations

import numpy as np

from .backend import Backend, resolve_backend
from .bfs import distance_maps
from .grid import MOVES, Cell, Grid
from .reservations import ReservationTable

__all__ = ["batched_astar", "space_time_astar"]

Path = list[Cell]


def batched_astar(
    grid: Grid,
    starts: np.ndarray,
    goals: np.ndarray,
    backend: Backend = "auto",
) -> list[Path | None]:
    """Solve many single-agent shortest-path queries in one batch.

    Args:
        grid: the occupancy grid.
        starts: ``(n, 2)`` start cells.
        goals: ``(n, 2)`` goal cells.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Returns:
        A list of ``n`` paths (lists of cells, start to goal inclusive),
        with ``None`` where the goal is unreachable from the start.
        Each returned path is cost-optimal.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=np.int32))
    goals = np.atleast_2d(np.asarray(goals, dtype=np.int32))
    if starts.shape != goals.shape:
        raise ValueError("starts and goals must have the same shape")
    # One BFS per goal: dist[i] is the exact cost-to-go for query i,
    # which doubles as the heuristic table other solvers reuse.
    dist = distance_maps(grid, goals, backend=backend)
    return [
        _descend(grid, dist[i], tuple(starts[i]), tuple(goals[i]))
        for i in range(len(starts))
    ]


def _descend(
    grid: Grid, dist: np.ndarray, start: Cell, goal: Cell
) -> Path | None:
    """Extract one path by walking the distance map downhill."""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    d = int(dist[start])
    if d < 0:
        return None
    path: Path = [start]
    cell = start
    while cell != goal:
        r, c = cell
        for dr, dc in MOVES[:4]:
            nxt = (r + int(dr), c + int(dc))
            if grid.in_bounds(nxt) and dist[nxt] == d - 1:
                cell = nxt
                break
        else:  # pragma: no cover - dist maps are consistent by construction
            raise AssertionError("inconsistent distance map")
        d -= 1
        path.append(cell)
    return path


def space_time_astar(
    grid: Grid,
    start: Cell,
    goal: Cell,
    table: ReservationTable | None = None,
    horizon: int | None = None,
    backend: Backend = "auto",
) -> Path | None:
    """Find a minimal-time path from ``start`` to ``goal`` under reservations.

    Waiting in place is a legal move. The agent may only settle on the
    goal after the last vertex reservation touching it, so a path is
    returned only when the agent can *stay* once it arrives — the same
    settle rule as pymapf's space-time A*.

    Args:
        grid: the occupancy grid.
        start: start cell.
        goal: goal cell.
        table: reservations to honour. When omitted, an empty table over
            ``horizon`` timesteps is used.
        horizon: last timestep considered. Defaults to the table's
            horizon, or ``4 * (height + width)`` for an empty table.
            The search is complete only up to this bound.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``. The table's array
            module must match the backend it is used with.

    Returns:
        ``path`` with ``path[t]`` the cell at time ``t``, ending on the
        goal, or ``None`` when no path exists within the horizon.
    """
    which = resolve_backend(backend)
    if table is None:
        default_h = horizon or 4 * (grid.height + grid.width)
        if which == "cuda":
            import cupy

            table = ReservationTable(grid, default_h, xp=cupy)
        else:
            table = ReservationTable(grid, default_h)
    horizon = min(horizon or table.horizon, table.horizon)
    if not grid.is_free(start) or not grid.is_free(goal):
        raise ValueError("start and goal must be free cells")
    if which == "cuda":
        return _space_time_cuda(grid, start, goal, table, horizon)
    return _space_time_cpu(grid, start, goal, table, horizon)


def _reconstruct(
    grid: Grid, step_from: np.ndarray, goal_linear: int, t: int
) -> Path:
    """Walk ``step_from`` back from ``(goal, t)`` to the start."""
    w = grid.width
    cells = [goal_linear]
    v = goal_linear
    for tau in range(t, 0, -1):
        move = int(step_from[tau, v])
        dr, dc = MOVES[move]
        v = v - int(dr) * w - int(dc)
        cells.append(v)
    cells.reverse()
    return [(v // w, v % w) for v in cells]


def _space_time_cpu(
    grid: Grid, start: Cell, goal: Cell, table: ReservationTable, horizon: int
) -> Path | None:
    if table.xp is not np:
        raise ValueError("CPU backend needs a NumPy-backed ReservationTable")
    h, w = grid.height, grid.width
    cells = h * w
    free = grid.free.reshape(-1)
    linear_index = np.arange(cells, dtype=np.int32)
    settle = table.last_vertex_time(goal)
    goal_linear = goal[0] * w + goal[1]
    start_linear = start[0] * w + start[1]

    reach = np.zeros(cells, dtype=bool)
    if table.vertex[0, start_linear]:
        return None
    reach[start_linear] = True
    step_from = np.full((horizon + 1, cells), -1, dtype=np.int8)

    if goal_linear == start_linear and settle < 0:
        return [start]

    reach2d = reach.reshape(h, w)
    for t in range(horizon):
        blocked = table.vertex[t + 1].astype(bool)
        arrived = table.arrived_from[t + 1]
        assigned = np.zeros(cells, dtype=bool)
        sf = step_from[t + 1]
        for move, (dr, dc) in enumerate(MOVES):
            # Cells v reachable via this move have predecessor u = v - d.
            shifted = _shift(reach2d, int(dr), int(dc)).reshape(-1)
            cand = shifted & free & ~blocked & ~assigned
            if move < 4 and cand.any():
                # Swap check: someone arrives at u coming from v, i.e.
                # arrived_from[u] == v. Shift arrived_from so position v
                # holds the value at its predecessor u.
                arrived_at_pred = _shift_int(
                    arrived.reshape(h, w), int(dr), int(dc)
                ).reshape(-1)
                cand &= arrived_at_pred != linear_index
            if cand.any():
                sf[cand] = move
                assigned |= cand
        if not assigned.any():
            return None
        reach2d = assigned.reshape(h, w)
        if assigned[goal_linear] and t + 1 > settle:
            return _reconstruct(grid, step_from, goal_linear, t + 1)
    return None


def _space_time_cuda(
    grid: Grid, start: Cell, goal: Cell, table: ReservationTable, horizon: int
) -> Path | None:
    import cupy

    from .kernels import get_kernel

    if table.xp is not cupy:
        raise ValueError("CUDA backend needs a CuPy-backed ReservationTable")
    kernel = get_kernel("spacetime", "spacetime_wave")
    h, w = grid.height, grid.width
    cells = h * w
    free_mask = cupy.asarray(grid.free.reshape(-1).astype(np.uint8))
    goal_linear = goal[0] * w + goal[1]
    start_linear = start[0] * w + start[1]
    settle = table.last_vertex_time(goal)

    if int(table.vertex[0, start_linear]):
        return None
    if goal_linear == start_linear and settle < 0:
        return [start]

    reach = cupy.zeros(cells, dtype=cupy.uint8)
    reach[start_linear] = 1
    reach_next = cupy.zeros(cells, dtype=cupy.uint8)
    step_from = cupy.full((horizon + 1, cells), -1, dtype=cupy.int8)
    changed = cupy.zeros(1, dtype=cupy.int32)

    threads = 256
    blocks = min(65535, (cells + threads - 1) // threads)
    for t in range(horizon):
        changed[0] = 0
        kernel(
            (blocks,),
            (threads,),
            (
                reach,
                reach_next,
                step_from[t + 1],
                free_mask,
                table.vertex[t + 1],
                table.arrived_from[t + 1],
                np.int32(h),
                np.int32(w),
                changed,
            ),
        )
        if int(changed[0]) == 0:
            return None
        if t + 1 > settle and int(reach_next[goal_linear]):
            return _reconstruct(
                grid, cupy.asnumpy(step_from[: t + 2]), goal_linear, t + 1
            )
        reach, reach_next = reach_next, reach
    return None


def _shift(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a 2D bool array by ``(dr, dc)``, filling with False."""
    if dr == 0 and dc == 0:
        return array
    out = np.zeros_like(array)
    h, w = array.shape
    rs = slice(max(dr, 0), h + min(dr, 0))
    cs = slice(max(dc, 0), w + min(dc, 0))
    rs_src = slice(max(-dr, 0), h + min(-dr, 0))
    cs_src = slice(max(-dc, 0), w + min(-dc, 0))
    out[rs, cs] = array[rs_src, cs_src]
    return out


def _shift_int(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a 2D int array by ``(dr, dc)``, filling with -2 (no match)."""
    out = np.full_like(array, -2)
    h, w = array.shape
    rs = slice(max(dr, 0), h + min(dr, 0))
    cs = slice(max(dc, 0), w + min(dc, 0))
    rs_src = slice(max(-dr, 0), h + min(-dr, 0))
    cs_src = slice(max(-dc, 0), w + min(-dc, 0))
    out[rs, cs] = array[rs_src, cs_src]
    return out
