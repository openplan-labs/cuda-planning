"""Prioritized planning and PIBT: validity, and parity with pymapf."""

import numpy as np
import pytest

from cuplan import PIBT, Grid, PrioritizedPlanning
from cuplan.benchmark.scenarios import random_scenario
from cuplan.problem import Agent, Problem


def small_problem():
    grid = Grid(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    agents = [
        Agent("a", (0, 0), (4, 4)),
        Agent("b", (4, 4), (0, 0)),
        Agent("c", (0, 4), (4, 0)),
    ]
    return Problem(grid, agents)


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanning, PIBT])
def test_solves_small_instance(solver_cls, backend):
    solution = solver_cls(backend=backend).solve(small_problem())
    assert solution is not None
    assert solution.is_valid()
    for agent in small_problem().agents:
        assert solution.paths[agent.name][0] == agent.start
        assert solution.paths[agent.name][-1] == agent.goal


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanning, PIBT])
@pytest.mark.parametrize("seed", [0, 1])
def test_random_instances_valid(solver_cls, backend, seed):
    scenario = random_scenario(16, 12, obstacle_density=0.15, seed=seed)
    solution = solver_cls(backend=backend).solve(scenario.to_cuplan())
    assert solution is not None
    assert solution.is_valid()


def test_prioritized_swap_corridor(backend):
    # Two agents facing each other in a ring: requires yielding.
    grid = Grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    problem = Problem(
        grid, [Agent("a", (0, 0), (0, 2)), Agent("b", (0, 2), (0, 0))]
    )
    solution = PrioritizedPlanning(backend=backend).solve(problem)
    assert solution is not None and solution.is_valid()


def test_unsolvable_disconnected(backend):
    grid = Grid([[0, 1, 0]])
    problem = Problem(grid, [Agent("a", (0, 0), (0, 2))])
    assert PrioritizedPlanning(backend=backend).solve(problem) is None
    assert PIBT(backend=backend).solve(problem) is None


def test_priority_order_respected():
    problem = small_problem()
    solution = PrioritizedPlanning(
        priority=["c", "b", "a"], backend="cpu"
    ).solve(problem)
    assert solution is not None and solution.is_valid()
    with pytest.raises(ValueError):
        PrioritizedPlanning(priority=["a"], backend="cpu").solve(problem)


def test_pibt_is_reproducible(backend):
    scenario = random_scenario(12, 8, seed=3)
    a = PIBT(seed=42, backend=backend).solve(scenario.to_cuplan())
    b = PIBT(seed=42, backend=backend).solve(scenario.to_cuplan())
    assert a is not None and b is not None
    assert a.paths == b.paths


def test_agrees_with_pymapf_on_prioritized():
    """Same instances, both libraries: valid plans, matching semantics.

    Exact sum-of-costs equality is not an invariant of prioritized
    planning — tie-breaking inside one agent's search changes the
    reservations later agents face — so the test pins what *is*
    invariant: both solve, both plans are conflict-free, every path
    respects the BFS lower bound, and the first-priority agent (which
    plans against an empty table) is exactly optimal in both.
    """
    pytest.importorskip("pymapf")
    import pymapf.algorithms  # noqa: F401
    from pymapf.core.solver import get_solver

    from cuplan import distance_maps

    for seed in (0, 1, 2):
        scenario = random_scenario(12, 6, seed=seed)
        ours = PrioritizedPlanning(backend="cpu").solve(scenario.to_cuplan())
        theirs = get_solver("prioritized").solve(scenario.to_pymapf())
        assert ours is not None and ours.is_valid()
        assert theirs is not None and theirs.is_valid()
        dist = distance_maps(
            scenario.grid, np.asarray(scenario.goals), backend="cpu"
        )
        for i in range(scenario.n_agents):
            lower = dist[i][scenario.starts[i]]
            assert len(ours.paths[f"a{i}"]) - 1 >= lower
            assert len(theirs.paths[f"a{i}"]) - 1 >= lower
        first = int(dist[0][scenario.starts[0]])
        assert len(ours.paths["a0"]) - 1 == first
        assert len(theirs.paths["a0"]) - 1 == first
