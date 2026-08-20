import numpy as np
import pytest

from cuplan import FlockingParams, FlockingSim
from cuplan.backend import cuda_available


def random_swarm(n=30, seed=0, dim=2):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-5, 5, (n, dim))
    velocities = rng.uniform(-1, 1, (n, dim))
    return positions, velocities


def test_flock_aligns_over_time(backend):
    positions, velocities = random_swarm()
    sim = FlockingSim(positions, velocities, backend=backend)
    result = sim.run(300)
    polar = result.polarization()
    # A jostling Boids flock does not reach perfect consensus — the
    # separation term keeps stirring it — but heading agreement must
    # rise substantially and settle well above the random start.
    assert polar[-50:].mean() > 0.6
    assert polar[-50:].mean() > polar[0] + 0.25


def test_flock_stays_cohesive(backend):
    positions, velocities = random_swarm(seed=2)
    sim = FlockingSim(positions, velocities, backend=backend)
    result = sim.run(300)
    spread = result.mean_neighbor_distance()
    params = sim.params
    assert spread[-1] < 3 * params.perception_radius


def test_speed_limit_respected(backend):
    positions, velocities = random_swarm(seed=3)
    params = FlockingParams(max_speed=1.5)
    sim = FlockingSim(positions, velocities, params=params, backend=backend)
    result = sim.run(100)
    speeds = np.linalg.norm(result.velocities, axis=-1)
    assert speeds.max() <= params.max_speed + 1e-9


@pytest.mark.skipif(not cuda_available(), reason="no working CUDA device")
def test_cuda_matches_cpu_forces():
    positions, velocities = random_swarm(n=50, seed=7)
    results = {}
    for backend in ("cpu", "cuda"):
        sim = FlockingSim(positions, velocities, backend=backend)
        results[backend] = sim.run(50)
    np.testing.assert_allclose(
        results["cpu"].positions, results["cuda"].positions, atol=1e-8
    )


def test_shape_validation():
    with pytest.raises(ValueError):
        FlockingSim(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        FlockingSim(np.zeros((3, 5)), np.zeros((3, 5)))
