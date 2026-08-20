"""Problem and solution types, mirroring pymapf's vocabulary.

``Agent``/``Problem``/``Solution`` carry the same semantics as
``pymapf.core.solver``: paths are lists of cells where index ``t`` is the
position at timestep ``t``, an agent parks on its goal after arrival,
and validity means no vertex conflict (two agents on one cell) and no
edge conflict (two agents swapping cells between ``t`` and ``t+1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Cell, Grid

Path = list[Cell]


@dataclass(frozen=True)
class Agent:
    """A planning agent: a unique ``name`` with ``start`` and ``goal`` cells."""

    name: str
    start: Cell
    goal: Cell


@dataclass
class Problem:
    """A multi-agent path finding instance on a 4-connected grid."""

    grid: Grid
    agents: list[Agent]

    def __post_init__(self) -> None:
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")
        for agent in self.agents:
            if not self.grid.is_free(agent.start):
                raise ValueError(
                    f"agent {agent.name!r} start {agent.start} is blocked "
                    "or out of bounds"
                )
            if not self.grid.is_free(agent.goal):
                raise ValueError(
                    f"agent {agent.name!r} goal {agent.goal} is blocked "
                    "or out of bounds"
                )

    @property
    def starts(self) -> np.ndarray:
        """``(n_agents, 2)`` start cells, in agent order."""
        return np.array([a.start for a in self.agents], dtype=np.int32)

    @property
    def goals(self) -> np.ndarray:
        """``(n_agents, 2)`` goal cells, in agent order."""
        return np.array([a.goal for a in self.agents], dtype=np.int32)


@dataclass(frozen=True)
class Conflict:
    """A vertex or edge conflict between two agents in a joint plan."""

    kind: str  # "vertex" or "edge"
    a: str
    b: str
    t: int
    cell_a: Cell
    cell_b: Cell  # equals cell_a for vertex conflicts


@dataclass
class Solution:
    """Result of a solve: one path per agent plus cost metrics.

    ``paths[name][t]`` is the agent's cell at timestep ``t``; index 0 is
    the start and the agent stays on its goal after the path ends.
    """

    paths: dict[str, Path]
    algorithm: str = ""
    backend: str = ""
    runtime: float = 0.0  # wall-clock seconds spent in ``solve``
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def makespan(self) -> int:
        """Timestep at which the last agent settles on its goal."""
        return max((len(p) - 1 for p in self.paths.values()), default=0)

    @property
    def sum_of_costs(self) -> int:
        """Sum over agents of the time spent before settling on the goal."""
        return sum(len(p) - 1 for p in self.paths.values())

    def first_conflict(self) -> Conflict | None:
        """Return the earliest conflict, or None for a valid plan."""
        return find_first_conflict(self.paths)

    def is_valid(self) -> bool:
        """True when the joint plan has no vertex or edge conflict."""
        return self.first_conflict() is None


def _cell_at(path: Path, t: int) -> Cell:
    return path[t] if t < len(path) else path[-1]


def find_first_conflict(paths: dict[str, Path]) -> Conflict | None:
    """Return the earliest vertex or edge conflict between any agent pair.

    Paths are implicitly padded: an agent that has arrived occupies its
    goal at every later timestep, exactly as in pymapf.

    Vectorized over agent pairs per timestep, so validating a
    500-agent plan costs milliseconds rather than the O(n^2 T) Python
    loop it replaces.
    """
    names = list(paths)
    n = len(names)
    if n < 2:
        return None
    horizon = max(len(p) for p in paths.values())
    # positions[t, i] = linear-ish tuple array; use structured comparison.
    pos = np.empty((horizon + 1, n, 2), dtype=np.int64)
    for i, name in enumerate(names):
        p = np.asarray(paths[name], dtype=np.int64)
        pos[: len(p), i] = p
        pos[len(p) :, i] = p[-1]
    # Encode cells as single integers for fast pairwise comparison.
    width = int(pos[..., 1].max()) + 2
    code = pos[..., 0] * width + pos[..., 1]  # (horizon+1, n)
    for t in range(horizon):
        now, nxt = code[t], code[t + 1]
        # Vertex conflicts at time t: duplicate codes.
        order = np.argsort(now, kind="stable")
        dup = now[order][:-1] == now[order][1:]
        if dup.any():
            k = int(np.argmax(dup))
            i, j = sorted((int(order[k]), int(order[k + 1])))
            cell = tuple(int(x) for x in pos[t, i])
            return Conflict("vertex", names[i], names[j], t, cell, cell)
        # Edge conflicts between t and t+1: i and j swap cells.
        swap = (now[:, None] == nxt[None, :]) & (nxt[:, None] == now[None, :])
        np.fill_diagonal(swap, False)
        if swap.any():
            i, j = np.argwhere(swap)[0]
            i, j = int(min(i, j)), int(max(i, j))
            return Conflict(
                "edge",
                names[i],
                names[j],
                t + 1,
                tuple(int(x) for x in pos[t + 1, i]),
                tuple(int(x) for x in pos[t + 1, j]),
            )
    # Final-timestep vertex conflicts (parked agents sharing a goal).
    now = code[horizon]
    order = np.argsort(now, kind="stable")
    dup = now[order][:-1] == now[order][1:]
    if dup.any():
        k = int(np.argmax(dup))
        i, j = sorted((int(order[k]), int(order[k + 1])))
        cell = tuple(int(x) for x in pos[horizon, i])
        return Conflict("vertex", names[i], names[j], horizon, cell, cell)
    return None


def paths_from_array(
    steps: np.ndarray, names: list[str], goals: np.ndarray
) -> dict[str, Path]:
    """Convert a ``(T+1, n, 2)`` position array to per-agent paths.

    The parked tail an agent spends on its goal is trimmed, matching
    pymapf's sum-of-costs convention (waiting on the goal at the end of
    a plan costs nothing).
    """
    paths: dict[str, Path] = {}
    for i, name in enumerate(names):
        cells: Path = [tuple(int(x) for x in cell) for cell in steps[:, i]]
        goal = tuple(int(x) for x in goals[i])
        end = len(cells) - 1
        while end > 0 and cells[end] == goal and cells[end - 1] == goal:
            end -= 1
        paths[name] = cells[: end + 1]
    return paths
