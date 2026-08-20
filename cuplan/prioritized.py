"""Prioritized planning (cooperative A*) with a device-resident table.

Agents are planned one at a time in priority order, each treating the
already-planned agents as moving obstacles (Erdmann and Lozano-Perez
1987; Silver 2005). The priority loop is inherently sequential — that is
the algorithm — so it stays on the host. What moves to the GPU is
everything inside one iteration: the constrained space-time search runs
as a frontier-parallel wavefront, and the reservation table lives on the
device the whole solve, so planning agent ``k`` never copies the
``k - 1`` previous paths back and forth.

Prioritized planning is *incomplete*: a bad priority order can fail on
solvable instances (Ma et al. 2019), and the bounded horizon adds a
second source of incompleteness that :class:`PrioritizedPlanning`
documents rather than hides.
"""

from __future__ import annotations

import time

import numpy as np

from .astar import space_time_astar
from .backend import Backend, resolve_backend
from .bfs import distance_maps
from .problem import Problem, Solution
from .reservations import ReservationTable

__all__ = ["PrioritizedPlanning"]


class PrioritizedPlanning:
    """Plan agents sequentially, reserving space-time cells as we go.

    Args:
        priority: optional list of agent names giving the planning
            order. Defaults to the order agents appear in the problem.
        horizon: last timestep considered per agent. Defaults to
            ``2 * max_goal_distance + 2 * n_agents + 16``, which gives a
            low-priority agent room to wait for everyone ahead of it on
            the instances this library targets. Raise it if solvable
            instances report failure.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``.
    """

    name = "prioritized"

    def __init__(
        self,
        priority: list[str] | None = None,
        horizon: int | None = None,
        backend: Backend = "auto",
    ):
        self.priority = priority
        self.horizon = horizon
        self.backend = backend

    def solve(self, problem: Problem) -> Solution | None:
        """Return a conflict-free :class:`~cuplan.problem.Solution` or None."""
        started = time.perf_counter()
        which = resolve_backend(self.backend)
        agents = {a.name: a for a in problem.agents}
        order = self.priority or [a.name for a in problem.agents]
        if set(order) != set(agents):
            raise ValueError("priority must list exactly the problem's agents")

        # Batched BFS from every goal: solvability check + horizon bound.
        dists = distance_maps(problem.grid, problem.goals, backend=which)
        starts = problem.starts
        goal_dist = dists[np.arange(len(order)), starts[:, 0], starts[:, 1]]
        if (goal_dist < 0).any():
            return None  # some agent cannot reach its goal at all
        horizon = self.horizon or int(
            2 * goal_dist.max() + 2 * len(order) + 16
        )

        if which == "cuda":
            import cupy as xp
        else:
            xp = np
        table = ReservationTable(problem.grid, horizon, xp=xp)

        paths = {}
        for name in order:
            agent = agents[name]
            path = space_time_astar(
                problem.grid,
                agent.start,
                agent.goal,
                table=table,
                horizon=horizon,
                backend=which,
            )
            if path is None:
                return None
            table.reserve_path(path)
            paths[name] = path

        return Solution(
            paths=paths,
            algorithm=self.name,
            backend=which,
            runtime=time.perf_counter() - started,
        )
