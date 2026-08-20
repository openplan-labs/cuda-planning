"""Batched BFS: CPU reference vs naive Python BFS, and CUDA vs CPU."""

from collections import deque

import numpy as np
import pytest

from cuplan import Grid, distance_maps
from cuplan.backend import cuda_available


def naive_bfs(grid: Grid, source):
    dist = np.full((grid.height, grid.width), -1, dtype=np.int32)
    dist[source] = 0
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for n in grid.neighbors(cell):
            if dist[n] < 0:
                dist[n] = dist[cell] + 1
                queue.append(n)
    return dist


def random_grid(size, density, seed):
    rng = np.random.default_rng(seed)
    obstacles = rng.random((size, size)) < density
    obstacles[0, 0] = False
    return Grid(obstacles)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpu_matches_naive(seed):
    grid = random_grid(24, 0.25, seed)
    free = np.argwhere(grid.free)
    rng = np.random.default_rng(seed)
    sources = free[rng.choice(len(free), size=5, replace=False)]
    dist = distance_maps(grid, sources, backend="cpu")
    for i, source in enumerate(sources):
        expected = naive_bfs(grid, tuple(source))
        assert (dist[i] == expected).all()


def test_unreachable_cells_are_minus_one():
    grid = Grid([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    dist = distance_maps(grid, [(0, 0)], backend="cpu")[0]
    assert dist[0, 2] == -1 and dist[1, 1] == -1
    assert dist[2, 0] == 2


def test_rejects_blocked_source():
    grid = Grid([[0, 1], [0, 0]])
    with pytest.raises(ValueError):
        distance_maps(grid, [(0, 1)], backend="cpu")


@pytest.mark.skipif(not cuda_available(), reason="no working CUDA device")
@pytest.mark.parametrize("seed", [0, 3])
def test_cuda_matches_cpu(seed):
    grid = random_grid(48, 0.2, seed)
    free = np.argwhere(grid.free)
    rng = np.random.default_rng(seed)
    sources = free[rng.choice(len(free), size=16, replace=False)]
    cpu = distance_maps(grid, sources, backend="cpu")
    cuda = distance_maps(grid, sources, backend="cuda")
    assert (cpu == cuda).all()
