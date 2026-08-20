"""Occupancy-grid world shared by every solver in cuplan.

The array is the data structure: a boolean ``(height, width)`` occupancy
map, truthy where blocked, matching ``pymapf.core.grid.GridMap``
semantics — 4-connected moves, unit edge costs, one move per timestep —
so a scenario ported between the two libraries means the same problem.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Cell = tuple[int, int]

# 4-connected orthogonal moves plus "wait", in a fixed order shared with
# the CUDA kernels (see kernels/spacetime.cu). Index 4 is wait.
MOVES: np.ndarray = np.array(
    [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32
)


class Grid:
    """An immutable 4-connected occupancy grid.

    Args:
        obstacles: 2D array-like, truthy where a cell is blocked. Nested
            lists and NumPy arrays both work; the grid is copied and
            frozen.
    """

    def __init__(self, obstacles: Iterable[Iterable]):
        array = np.asarray(obstacles)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("obstacles must be a non-empty 2D array")
        self.obstacles: np.ndarray = array.astype(bool).copy()
        self.obstacles.setflags(write=False)
        self.height, self.width = self.obstacles.shape

    @classmethod
    def empty(cls, height: int, width: int) -> Grid:
        """Return an obstacle-free grid of the given shape."""
        return cls(np.zeros((height, width), dtype=bool))

    @property
    def free(self) -> np.ndarray:
        """Boolean map of traversable cells (the complement of obstacles)."""
        return ~self.obstacles

    @property
    def free_cells(self) -> int:
        """Number of traversable cells."""
        return int(self.free.sum())

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.obstacles[cell]

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Return the free, in-bounds 4-connected neighbours of ``cell``."""
        r, c = cell
        result = []
        for dr, dc in MOVES[:4]:
            n = (r + int(dr), c + int(dc))
            if self.is_free(n):
                result.append(n)
        return result

    def to_linear(self, cells: np.ndarray) -> np.ndarray:
        """Convert ``(..., 2)`` row/col coordinates to linear indices."""
        cells = np.asarray(cells)
        return cells[..., 0] * self.width + cells[..., 1]

    def from_linear(self, index: np.ndarray) -> np.ndarray:
        """Convert linear indices back to ``(..., 2)`` row/col pairs."""
        index = np.asarray(index)
        return np.stack([index // self.width, index % self.width], axis=-1)

    def __repr__(self) -> str:
        return (
            f"Grid(height={self.height}, width={self.width}, "
            f"obstacles={int(self.obstacles.sum())})"
        )
