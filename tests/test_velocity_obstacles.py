import numpy as np
import pytest

from cuplan import VelocityObstacleSim
from cuplan.backend import cuda_available


def two_agent_sim(backend):
    sim = VelocityObstacleSim(backend=backend)
    sim.add_agent((0.0, 0.0), (10.0, 0.0))
    sim.add_agent((10.0, 0.5), (0.0, 0.5))
    return sim


def test_agents_reach_goals(backend):
    sim = two_agent_sim(backend)
    result = sim.run(120)
    assert result.goals_reached(sim.goals, tolerance=0.5) == 2


def test_agents_do_not_collide(backend):
    sim = two_agent_sim(backend)
    result = sim.run(120)
    # 2.2 x radius is the planning margin; require physical separation.
    assert result.min_separation() > 2 * sim.radius * 0.9


def test_single_agent_goes_straight(backend):
    sim = VelocityObstacleSim(backend=backend)
    sim.add_agent((0.0, 0.0), (5.0, 0.0))
    result = sim.run(60)
    assert result.goals_reached(sim.goals, tolerance=0.3) == 1
    # No obstacles: y stays near zero throughout.
    assert np.abs(result.positions[:, 0, 1]).max() < 0.3


def test_moving_obstacle_is_avoided(backend):
    sim = VelocityObstacleSim(backend=backend)
    sim.add_agent((0.0, 0.0), (10.0, 0.0))
    sim.add_obstacle((10.0, 0.0), (-1.0, 0.0))
    result = sim.run(100)
    # Distance between the agent and the obstacle stays positive.
    for k, frame in enumerate(result.positions[:-1]):
        obstacle = np.array([10.0, 0.0]) + np.array([-1.0, 0.0]) * k * sim.timestep
        assert np.linalg.norm(frame[0] - obstacle) > sim.radius


@pytest.mark.skipif(not cuda_available(), reason="no working CUDA device")
def test_cuda_matches_cpu_trajectories():
    runs = {}
    for backend in ("cpu", "cuda"):
        sim = VelocityObstacleSim(backend=backend)
        rng = np.random.default_rng(5)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 8))
        starts = 4.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        for s in starts:
            sim.add_agent(s, -s)
        runs[backend] = sim.run(50)
    np.testing.assert_allclose(
        runs["cpu"].positions, runs["cuda"].positions, atol=1e-9
    )


def test_run_requires_agents():
    with pytest.raises(ValueError):
        VelocityObstacleSim(backend="cpu").run(5)
