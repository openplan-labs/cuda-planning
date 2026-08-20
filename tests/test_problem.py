import pytest

from cuplan import Grid, Solution, find_first_conflict
from cuplan.problem import Agent, Problem


def test_detects_vertex_conflict():
    conflict = find_first_conflict(
        {"a": [(0, 0), (0, 1)], "b": [(0, 2), (0, 1)]}
    )
    assert conflict is not None
    assert conflict.kind == "vertex" and conflict.t == 1


def test_detects_edge_conflict():
    conflict = find_first_conflict(
        {"a": [(0, 0), (0, 1)], "b": [(0, 1), (0, 0)]}
    )
    assert conflict is not None
    assert conflict.kind == "edge" and conflict.t == 1


def test_padding_parks_agents_on_goal():
    # b arrives early and parks; a runs into it later.
    conflict = find_first_conflict(
        {"a": [(0, 0), (0, 1), (0, 2)], "b": [(0, 2)]}
    )
    assert conflict is not None and conflict.kind == "vertex"


def test_valid_plan_has_no_conflict():
    paths = {"a": [(0, 0), (1, 0)], "b": [(0, 1), (0, 0)]}
    assert find_first_conflict(paths) is None
    solution = Solution(paths=paths)
    assert solution.is_valid()
    assert solution.sum_of_costs == 2
    assert solution.makespan == 1


def test_problem_validation():
    grid = Grid([[0, 1]])
    with pytest.raises(ValueError):
        Problem(grid, [Agent("a", (0, 0), (0, 1))])
    with pytest.raises(ValueError):
        Problem(
            grid,
            [Agent("a", (0, 0), (0, 0)), Agent("a", (0, 0), (0, 0))],
        )


def test_roadmap_stubs_raise():
    from cuplan import roadmap

    for name in roadmap.__all__:
        with pytest.raises(NotImplementedError):
            getattr(roadmap, name)()
