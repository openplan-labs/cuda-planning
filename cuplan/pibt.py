"""Priority Inheritance with Backtracking (PIBT), GPU-assisted.

PIBT (Okumura et al. 2022, *Artificial Intelligence* 310:103752) plans
one timestep at a time: each agent proposes the neighbouring vertex
closest to its goal, and conflicts are settled on the spot by priority
inheritance with backtracking. Priorities grow while an agent is away
from its goal and reset on arrival, which prevents starvation.

What parallelizes and what does not, stated plainly:

* The **distance oracle** — one exact goal-distance table per agent,
  the dominant cost in pymapf's implementation (one serial Dijkstra per
  agent) — is a single batched BFS on the selected backend.
* The **candidate evaluation** — gathering the five candidate vertices
  of every agent and ordering them by goal distance with random
  tie-breaking — is one vectorized gather + argsort across all agents
  per timestep.
* The **inheritance chains** are recursive and data-dependent, so they
  run on the host, exactly as written in the paper.

PIBT is *incomplete*: it can livelock on instances that require an
agent to move far away from its goal (LaCAM, on the roadmap, fixes
that by wrapping PIBT in a complete search). Reachability of every goal
is checked up front, so failures are reported as livelock, not silence.
"""

from __future__ import annotations

import time

import numpy as np

from .backend import Backend, resolve_backend
from .bfs import distance_maps
from .grid import MOVES, Grid
from .problem import Problem, Solution, paths_from_array

__all__ = ["PIBT"]

_INF = np.int64(1 << 40)


def _candidate_table(grid: Grid) -> np.ndarray:
    """Per-cell candidate moves: ``(cells, 5)`` linear indices, -1 invalid.

    Column order matches :data:`cuplan.grid.MOVES`; the last column
    (wait) is always the cell itself.
    """
    h, w = grid.height, grid.width
    r, c = np.mgrid[0:h, 0:w]
    table = np.full((h * w, 5), -1, dtype=np.int64)
    for m, (dr, dc) in enumerate(MOVES):
        nr, nc = r + int(dr), c + int(dc)
        valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        valid &= np.where(valid, grid.free[nr.clip(0, h - 1), nc.clip(0, w - 1)], False)
        table[:, m] = np.where(valid, nr * w + nc, -1).reshape(-1)
    table[:, 4] = (r * w + c).reshape(-1)  # waiting on a free cell is legal
    return table


class PIBT:
    """Rule-based one-step-at-a-time MAPF solver (Okumura et al. 2022).

    Args:
        max_timestep: give up after this many timesteps. Defaults to a
            bound proportional to the map size and the agent count.
        seed: fixes tie-breaking, making a run reproducible.
        backend: backend used for the batched distance oracle. The
            per-step candidate ranking and the inheritance chains run on
            the host in both backends, so this changes how the distance
            maps are computed and nothing else.
    """

    name = "pibt"

    def __init__(
        self,
        max_timestep: int | None = None,
        seed: int | None = 0,
        backend: Backend = "auto",
    ):
        self.max_timestep = max_timestep
        self.seed = seed
        self.backend = backend

    def solve(self, problem: Problem) -> Solution | None:
        """Return a conflict-free :class:`~cuplan.problem.Solution` or None."""
        started = time.perf_counter()
        which = resolve_backend(self.backend)
        grid = problem.grid
        w = grid.width
        n = len(problem.agents)
        names = [a.name for a in problem.agents]
        rng = np.random.default_rng(self.seed)

        # Batched exact distance oracle: (n, cells), -1 -> +inf.
        dist = (
            distance_maps(grid, problem.goals, backend=which)
            .reshape(n, -1)
            .astype(np.int64)
        )
        dist[dist < 0] = _INF

        positions = problem.starts[:, 0].astype(np.int64) * w + problem.starts[:, 1]
        goals = problem.goals[:, 0].astype(np.int64) * w + problem.goals[:, 1]
        if (dist[np.arange(n), positions] >= _INF).any():
            return None  # some agent cannot reach its goal
        candidates = _candidate_table(grid)

        base = np.arange(n, dtype=np.float64) / (n + 1)
        priorities = base.copy()
        horizon = self.max_timestep or (grid.free_cells + 4 * n + 8)

        steps = [positions.copy()]
        for _ in range(horizon):
            if (positions == goals).all():
                break
            order = np.argsort(-priorities, kind="stable")
            ranked = self._rank_candidates(dist, candidates, positions, rng)
            nxt = _pibt_step(positions, ranked, order)
            if nxt is None:
                return None
            positions = nxt
            steps.append(positions.copy())
            at_goal = positions == goals
            priorities = np.where(at_goal, base, priorities + 1.0)

        if not (positions == goals).all():
            return None  # livelock within the horizon

        array = np.stack(steps)  # (T+1, n)
        cells = np.stack([array // w, array % w], axis=-1)
        return Solution(
            paths=paths_from_array(cells, names, problem.goals),
            algorithm=self.name,
            backend=which,
            runtime=time.perf_counter() - started,
        )

    def _rank_candidates(
        self,
        dist: np.ndarray,
        candidates: np.ndarray,
        positions: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Order every agent's candidate vertices by goal distance.

        One gather + argsort over the whole batch: ``(n, 5)`` candidate
        cells, distances looked up in each agent's own table, ties
        broken by a fresh random key (the randomness LaCAM relies on).
        Invalid candidates sort last and are marked -1.
        """
        n = len(positions)
        cand = candidates[positions]  # (n, 5)
        valid = cand >= 0
        d = np.where(valid, dist[np.arange(n)[:, None], cand.clip(min=0)], _INF)
        d = np.where(d >= _INF, _INF, d)
        keys = d.astype(np.float64) + rng.random((n, 5))
        keys[~valid] = np.inf
        order = np.argsort(keys, axis=1, kind="stable")
        ranked = np.take_along_axis(cand, order, axis=1)
        ranked[np.take_along_axis(~valid, order, axis=1)] = -1
        return ranked


def _pibt_step(
    positions: np.ndarray, ranked: np.ndarray, order: np.ndarray
) -> np.ndarray | None:
    """One PIBT timestep: recursive priority inheritance on the host.

    ``ranked[i]`` lists agent ``i``'s candidate vertices best-first
    (-1 padding). Returns the next configuration or None when no valid
    configuration exists from here.
    """
    n = len(positions)
    occupied_now: dict[int, int] = {int(p): i for i, p in enumerate(positions)}
    occupied_next: dict[int, int] = {}
    decided = np.full(n, -1, dtype=np.int64)
    journal: list[int] = []

    def assign(agent: int, cell: int) -> None:
        occupied_next[cell] = agent
        decided[agent] = cell
        journal.append(agent)

    def rollback(mark: int) -> None:
        while len(journal) > mark:
            agent = journal.pop()
            cell = int(decided[agent])
            decided[agent] = -1
            if occupied_next.get(cell) == agent:
                del occupied_next[cell]

    def funnel(agent: int, higher: int | None) -> bool:
        current = int(positions[agent])
        for cell in ranked[agent]:
            cell = int(cell)
            if cell < 0 or cell in occupied_next:
                continue
            # Refuse a head-on swap with the agent that called us.
            if higher is not None and cell == int(positions[higher]):
                continue
            other = occupied_now.get(cell)
            # ...and with anyone already committed to our vertex.
            if other is not None and decided[other] == current:
                continue
            mark = len(journal)
            assign(agent, cell)
            if other is not None and other != agent and decided[other] < 0:
                if not funnel(other, agent):
                    rollback(mark)
                    continue
            return True
        # Nowhere to go: stay put if our own vertex is still free.
        if current not in occupied_next:
            assign(agent, current)
            return True
        return False

    for agent in order:
        agent = int(agent)
        if decided[agent] < 0 and not funnel(agent, None):
            return None
    if (decided < 0).any():
        return None
    # Final proof, as in pymapf: distinct vertices and no pair swapping.
    if len(set(decided.tolist())) != n:
        return None
    for agent in range(n):
        other = occupied_now.get(int(decided[agent]))
        if (
            other is not None
            and other != agent
            and decided[other] == positions[agent]
        ):
            return None
    return decided.copy()
