"""Boids flocking with GPU force accumulation.

Mirrors ``pymapf.swarm.flocking.Boids`` (Reynolds 1987): separation as
an inverse-square repulsion inside the separation distance, cohesion
toward the mean neighbour offset, alignment toward the mean neighbour
velocity, all limited to a maximum acceleration and integrated at a
capped speed.

The per-agent force is a sum over neighbours — a gather with no
data dependencies between agents — so the CUDA backend runs one thread
per agent scanning the swarm. The scan is brute-force O(n^2) on both
backends on purpose: at the swarm sizes this library targets (up to a
few thousand agents) rebuilding a spatial index every step costs more
than it saves, and the two backends stay exactly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend, resolve_backend

__all__ = ["FlockingParams", "FlockingSim", "FlockingResult"]


@dataclass(frozen=True)
class FlockingParams:
    """Boids gains and limits, defaults matching pymapf's ``Boids``."""

    separation_gain: float = 6.0
    cohesion_gain: float = 1.2
    alignment_gain: float = 2.5
    perception_radius: float = 8.0
    separation_distance: float = 1.5
    max_accel: float = 4.0
    max_speed: float = 2.5


@dataclass
class FlockingResult:
    """Trajectories and metrics of a flocking run."""

    positions: np.ndarray  # (T+1, n, dim)
    velocities: np.ndarray  # (T+1, n, dim)
    runtime: float
    backend: str
    extra: dict = field(default_factory=dict)

    def polarization(self) -> np.ndarray:
        """Per-frame heading agreement: 1.0 means perfectly aligned.

        The norm of the mean unit velocity (Vicsek's order parameter).
        """
        v = self.velocities
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        unit = np.divide(v, norm, out=np.zeros_like(v), where=norm > 1e-12)
        return np.linalg.norm(unit.mean(axis=1), axis=-1)

    def mean_neighbor_distance(self) -> np.ndarray:
        """Per-frame mean pairwise distance (a cohesion proxy)."""
        out = []
        n = self.positions.shape[1]
        iu = np.triu_indices(n, k=1)
        for frame in self.positions:
            diff = frame[:, None, :] - frame[None, :, :]
            out.append(float(np.linalg.norm(diff, axis=-1)[iu].mean()))
        return np.asarray(out)


class FlockingSim:
    """Boids swarm simulation (Reynolds 1987).

    Args:
        positions: ``(n, dim)`` initial positions, dim in {2, 3}.
        velocities: ``(n, dim)`` initial velocities.
        params: gains and limits; see :class:`FlockingParams`.
        timestep: integration step in seconds.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        params: FlockingParams | None = None,
        timestep: float = 0.05,
        backend: Backend = "auto",
    ):
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.velocities = np.asarray(velocities, dtype=np.float64).copy()
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must share a shape")
        if self.positions.ndim != 2 or self.positions.shape[1] not in (2, 3):
            raise ValueError("positions must be (n, 2) or (n, 3)")
        self.params = params or FlockingParams()
        self.timestep = float(timestep)
        self.backend = backend

    def run(self, n_steps: int) -> FlockingResult:
        """Integrate ``n_steps`` and return trajectories plus metrics."""
        started = time.perf_counter()
        which = resolve_backend(self.backend)
        pos = self.positions.copy()
        vel = self.velocities.copy()
        p = self.params

        if which == "cuda":
            forces = self._make_cuda_forces()
        else:
            forces = self._forces_cpu

        history_p = [pos.copy()]
        history_v = [vel.copy()]
        for _ in range(n_steps):
            command = forces(pos, vel)
            vel = vel + command * self.timestep
            speed = np.linalg.norm(vel, axis=1, keepdims=True)
            over = speed[:, 0] > p.max_speed
            vel[over] *= (p.max_speed / speed[over])
            pos = pos + vel * self.timestep
            history_p.append(pos.copy())
            history_v.append(vel.copy())

        return FlockingResult(
            positions=np.stack(history_p),
            velocities=np.stack(history_v),
            runtime=time.perf_counter() - started,
            backend=which,
        )

    def _forces_cpu(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Vectorized full-pairwise Boids forces."""
        p = self.params
        n, dim = pos.shape
        offsets = pos[None, :, :] - pos[:, None, :]  # (i, j, d): j - i
        dist = np.linalg.norm(offsets, axis=-1)
        neighbour = (dist <= p.perception_radius) & ~np.eye(n, dtype=bool)
        counts = neighbour.sum(axis=1)

        safe = np.maximum(dist, 1e-6)
        close = neighbour & (dist < p.separation_distance)
        sep = -(offsets / safe[..., None] ** 2 * close[..., None]).sum(axis=1)
        coh = (offsets * neighbour[..., None]).sum(axis=1)
        ali = ((vel[None, :, :] - vel[:, None, :]) * neighbour[..., None]).sum(
            axis=1
        )
        denom = np.maximum(counts, 1)[:, None]
        command = (
            p.separation_gain * sep
            + p.cohesion_gain * coh / denom
            + p.alignment_gain * ali / denom
        )
        command[counts == 0] = 0.0
        norm = np.linalg.norm(command, axis=1, keepdims=True)
        over = norm[:, 0] > p.max_accel
        command[over] *= p.max_accel / norm[over]
        return command

    def _make_cuda_forces(self):
        import cupy

        from .kernels import get_kernel

        kernel = get_kernel("flocking", "boids_forces")
        p = self.params
        dim = self.positions.shape[1]

        def forces(pos, vel):
            n = len(pos)
            pos_d = cupy.asarray(pos)
            vel_d = cupy.asarray(vel)
            out = cupy.empty((n, dim), dtype=cupy.float64)
            threads = 128
            blocks = min(65535, (n + threads - 1) // threads)
            kernel(
                (blocks,),
                (threads,),
                (
                    pos_d,
                    vel_d,
                    out,
                    np.int32(n),
                    np.int32(dim),
                    np.float64(p.perception_radius),
                    np.float64(p.separation_distance),
                    np.float64(p.separation_gain),
                    np.float64(p.cohesion_gain),
                    np.float64(p.alignment_gain),
                    np.float64(p.max_accel),
                ),
            )
            return cupy.asnumpy(out)

        return forces
