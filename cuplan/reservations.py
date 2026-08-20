"""Space-time reservation table shared by the constrained searches.

The table is two dense arrays over ``(timestep, cell)``:

* ``vertex[t, v]`` — cell ``v`` is occupied at time ``t`` (vertex
  constraint, Silver 2005).
* ``arrived_from[t, v]`` — linear index of the cell the occupying agent
  came from, or ``-1``. Because vertex reservations guarantee at most
  one agent arrives at ``v`` per timestep, this single integer encodes
  every edge (swap) constraint: a move ``u -> v`` arriving at ``t`` is
  illegal exactly when ``arrived_from[t, u] == v``.

Dense arrays instead of pymapf's constraint sets is the whole trick:
membership tests become array lookups the wavefront can do for every
cell at once, on either backend (the ``xp`` module is NumPy or CuPy).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import Grid

__all__ = ["ReservationTable"]


class ReservationTable:
    """Dense vertex + edge reservations over a bounded time horizon.

    Args:
        grid: the occupancy grid the reservations refer to.
        horizon: last timestep (inclusive) the table covers. Searches
            against the table cannot return paths longer than this.
        xp (module): array module — ``numpy`` (default) or ``cupy``. Solvers
            running on the CUDA backend keep the table on the device so
            reserving a path never round-trips through host memory.
    """

    def __init__(self, grid: Grid, horizon: int, xp=np):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.grid = grid
        self.horizon = int(horizon)
        self.xp = xp
        cells = grid.height * grid.width
        self.vertex = xp.zeros((self.horizon + 1, cells), dtype=xp.uint8)
        self.arrived_from = xp.full(
            (self.horizon + 1, cells), -1, dtype=xp.int32
        )

    def reserve_path(self, path: Sequence[tuple]) -> None:
        """Reserve a full agent path, parking it on its last cell forever.

        ``path[t]`` is the agent's cell at time ``t``. After the path
        ends the agent is assumed to stay on its final cell, so that
        cell is blocked through the end of the horizon — the same
        convention as pymapf's prioritized planner.
        """
        xp = self.xp
        cells = np.asarray(path, dtype=np.int64)
        linear = cells[:, 0] * self.grid.width + cells[:, 1]
        if len(linear) > self.horizon + 1:
            raise ValueError("path is longer than the table horizon")
        steps = xp.asarray(linear)
        t = xp.arange(len(linear))
        self.vertex[t, steps] = 1
        # Park on the final cell for the rest of the horizon.
        self.vertex[len(linear) :, int(linear[-1])] = 1
        # Record arrivals for edge (swap) constraints.
        if len(linear) > 1:
            self.arrived_from[t[1:], steps[1:]] = steps[:-1].astype(xp.int32)

    def block_vertex(self, cell: tuple, t: int) -> None:
        """Add a single vertex constraint: ``cell`` is occupied at ``t``."""
        r, c = cell
        self.vertex[t, r * self.grid.width + c] = 1

    def block_edge(self, u: tuple, v: tuple, t: int) -> None:
        """Forbid traversing ``u -> v`` arriving at time ``t``."""
        w = self.grid.width
        self.arrived_from[t, u[0] * w + u[1]] = np.int32(v[0] * w + v[1])

    def last_vertex_time(self, cell: tuple) -> int:
        """Latest ``t`` with a vertex reservation on ``cell`` (-1 if none).

        A search must not settle on its goal before this time — the
        reservation could push it off again.
        """
        r, c = cell
        column = self.vertex[:, r * self.grid.width + c]
        hits = self.xp.flatnonzero(column)
        return int(hits[-1]) if len(hits) else -1
