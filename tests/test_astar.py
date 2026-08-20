"""Batched shortest paths and constrained space-time search."""

import numpy as np
import pytest

from cuplan import (
    Grid,
    ReservationTable,
    batched_astar,
    distance_maps,
    space_time_astar,
)
from cuplan.backend import cuda_available


def _valid_path(grid, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:], strict=False):
        assert grid.is_free(b)
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1


def test_batched_astar_optimal(backend):
    rng = np.random.default_rng(7)
    grid = Grid(rng.random((32, 32)) < 0.2)
    free = np.argwhere(grid.free)
    picks = free[rng.choice(len(free), size=20, replace=False)]
    starts, goals = picks[:10], picks[10:]
    paths = batched_astar(grid, starts, goals, backend=backend)
    dist = distance_maps(grid, goals, backend="cpu")
    for i, path in enumerate(paths):
        start = tuple(starts[i])
        goal = tuple(goals[i])
        optimal = dist[i][start]
        if optimal < 0:
            assert path is None
        else:
            _valid_path(grid, path, start, goal)
            assert len(path) - 1 == optimal


def test_space_time_waits_for_crossing_agent(backend):
    # A corridor: a vertex reservation forces a wait.
    grid = Grid([[0, 0, 0, 0]])
    table = _table(grid, 10, backend)
    table.block_vertex((0, 2), 2)  # someone sits on cell 2 at t=2
    path = space_time_astar(
        grid, (0, 0), (0, 3), table=table, backend=backend
    )
    assert path is not None
    assert path[-1] == (0, 3)
    # Cannot be on (0,2) at t=2.
    padded = path + [path[-1]] * 5
    assert padded[2] != (0, 2)
    assert len(path) - 1 >= 3


def test_space_time_edge_constraint_prevents_swap(backend):
    grid = Grid([[0, 0]])
    table = _table(grid, 8, backend)
    # A reserved agent moves (0,1) -> (0,0) arriving at t=1: swapping
    # against it is illegal, and its target cell is occupied at t=1.
    table.reserve_path([(0, 1), (0, 0)])
    path = space_time_astar(grid, (0, 0), (0, 1), table=table, backend=backend)
    # The only move is blocked by the swap and (0,0) is occupied from
    # t=1 on (the reserved agent parks), so no path exists.
    assert path is None


def test_space_time_respects_settle_rule(backend):
    grid = Grid([[0, 0, 0]])
    table = _table(grid, 12, backend)
    # Goal cell occupied at t=5: the agent may only settle after that.
    table.block_vertex((0, 2), 5)
    path = space_time_astar(grid, (0, 0), (0, 2), table=table, backend=backend)
    assert path is not None
    assert len(path) - 1 > 5


def test_space_time_start_equals_goal(backend):
    grid = Grid([[0, 0]])
    path = space_time_astar(grid, (0, 0), (0, 0), backend=backend)
    assert path == [(0, 0)]


def test_space_time_unreachable_returns_none(backend):
    grid = Grid([[0, 1, 0]])
    path = space_time_astar(grid, (0, 0), (0, 2), backend=backend)
    assert path is None


@pytest.mark.skipif(not cuda_available(), reason="no working CUDA device")
def test_space_time_cuda_matches_cpu_costs():
    rng = np.random.default_rng(11)
    grid = Grid(rng.random((16, 16)) < 0.2)
    free = np.argwhere(grid.free)
    picks = free[rng.choice(len(free), size=12, replace=False)]
    for i in range(6):
        start, goal = tuple(picks[i]), tuple(picks[i + 6])
        cpu = space_time_astar(grid, start, goal, backend="cpu")
        cuda = space_time_astar(grid, start, goal, backend="cuda")
        if cpu is None:
            assert cuda is None
        else:
            assert cuda is not None
            assert len(cpu) == len(cuda)


def _table(grid, horizon, backend):
    if backend == "cuda":
        import cupy

        return ReservationTable(grid, horizon, xp=cupy)
    return ReservationTable(grid, horizon)
