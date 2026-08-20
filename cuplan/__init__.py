"""cuplan: CUDA-accelerated multi-agent path finding and planning.

The GPU sibling of `pymapf <https://github.com/openplan-labs/pymapf>`_.
Every algorithm ships two implementations behind one API — a vectorized
NumPy reference and a CUDA realization compiled at runtime through CuPy's
NVRTC bindings — selected with ``backend="auto" | "cpu" | "cuda"``.

The GPU wins by batching, not by forcing serial searches onto device
threads: one distance map per agent, one wavefront per timestep, one
thread per (agent, velocity-sample) pair. Where a step is inherently
sequential (priority orders, priority inheritance) it stays on the host
and only its data-parallel core moves to the device — each module's
docstring says which part that is.
"""

from __future__ import annotations

from .astar import batched_astar, space_time_astar
from .backend import cuda_available, resolve_backend
from .bfs import distance_maps
from .flocking import FlockingParams, FlockingSim
from .grid import Grid
from .pibt import PIBT
from .prioritized import PrioritizedPlanning
from .problem import Agent, Problem, Solution, find_first_conflict
from .reservations import ReservationTable
from .velocity_obstacles import VelocityObstacleSim

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "FlockingParams",
    "FlockingSim",
    "Grid",
    "PIBT",
    "PrioritizedPlanning",
    "Problem",
    "ReservationTable",
    "Solution",
    "VelocityObstacleSim",
    "batched_astar",
    "cuda_available",
    "distance_maps",
    "find_first_conflict",
    "resolve_backend",
    "space_time_astar",
    "__version__",
]
