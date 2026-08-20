"""Reproducible random MAPF scenarios shared across libraries.

A scenario is a seeded random obstacle grid plus distinct start and
goal cells, all mutually reachable (verified with a flood fill from the
first start). The same object converts to a cuplan
:class:`~cuplan.problem.Problem` and a pymapf ``MAPFProblem``, which is
what makes the benchmark apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bfs import distance_maps
from ..grid import Grid
from ..problem import Agent, Problem

__all__ = ["Scenario", "random_scenario"]


@dataclass(frozen=True)
class Scenario:
    """A reproducible MAPF instance description."""

    grid: Grid
    starts: tuple[tuple[int, int], ...]
    goals: tuple[tuple[int, int], ...]
    seed: int

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    def to_cuplan(self) -> Problem:
        """Return the instance as a cuplan :class:`~cuplan.problem.Problem`."""
        agents = [
            Agent(name=f"a{i}", start=s, goal=g)
            for i, (s, g) in enumerate(zip(self.starts, self.goals, strict=True))
        ]
        return Problem(grid=self.grid, agents=agents)

    def to_pymapf(self):
        """Return the instance as a ``pymapf`` problem (imported lazily)."""
        from pymapf.core.grid import GridMap
        from pymapf.core.solver import Agent as PAgent
        from pymapf.core.solver import MAPFProblem

        grid = GridMap(self.grid.obstacles.astype(int).tolist())
        agents = [
            PAgent(name=f"a{i}", start=s, goal=g)
            for i, (s, g) in enumerate(zip(self.starts, self.goals, strict=True))
        ]
        return MAPFProblem(grid=grid, agents=agents)


def random_scenario(
    size: int,
    n_agents: int,
    obstacle_density: float = 0.15,
    seed: int = 0,
) -> Scenario:
    """Generate a connected random instance.

    Obstacles are sampled i.i.d. at ``obstacle_density``; starts and
    goals are distinct free cells drawn from the largest connected
    component, so every agent's goal is reachable.

    Args:
        size: grid is ``size x size``.
        n_agents: number of agents (must fit in the free space).
        obstacle_density: fraction of blocked cells.
        seed: RNG seed; same seed, same instance.
    """
    rng = np.random.default_rng(seed)
    for _attempt in range(64):
        obstacles = rng.random((size, size)) < obstacle_density
        grid = Grid(obstacles)
        free = np.argwhere(grid.free)
        if len(free) < 2 * n_agents:
            continue
        # Largest connected component via one flood fill per candidate seed.
        seed_cell = free[rng.integers(len(free))]
        dist = distance_maps(grid, seed_cell[None, :], backend="cpu")[0]
        component = np.argwhere(dist >= 0)
        if len(component) < 2 * n_agents:
            continue
        picks = rng.choice(len(component), size=2 * n_agents, replace=False)
        cells: list[tuple[int, int]] = [
            (int(r), int(c)) for r, c in component[picks]
        ]
        return Scenario(
            grid=grid,
            starts=tuple(cells[:n_agents]),
            goals=tuple(cells[n_agents:]),
            seed=seed,
        )
    raise RuntimeError(
        f"could not build a scenario with {n_agents} agents on a "
        f"{size}x{size} grid at density {obstacle_density}"
    )
