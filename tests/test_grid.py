import numpy as np
import pytest

from cuplan import Grid


def test_grid_shape_and_free_cells():
    grid = Grid([[0, 1], [0, 0]])
    assert (grid.height, grid.width) == (2, 2)
    assert grid.free_cells == 3
    assert grid.is_free((0, 0))
    assert not grid.is_free((0, 1))
    assert not grid.is_free((-1, 0))


def test_neighbors_are_free_and_in_bounds():
    grid = Grid([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    assert set(grid.neighbors((1, 1))) == {(1, 0), (1, 2)}
    assert set(grid.neighbors((0, 0))) == {(1, 0)}


def test_linear_roundtrip():
    grid = Grid.empty(3, 5)
    cells = np.array([[0, 0], [2, 4], [1, 3]])
    assert (grid.from_linear(grid.to_linear(cells)) == cells).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        Grid(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        Grid(np.zeros(4))


def test_grid_is_immutable():
    grid = Grid.empty(2, 2)
    with pytest.raises(ValueError):
        grid.obstacles[0, 0] = True
