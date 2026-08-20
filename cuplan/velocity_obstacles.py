"""Decentralized collision avoidance with velocity obstacles.

Mirrors ``pymapf.decentralized.velocity_obstacle`` (Fiorini and Shiller
1998): each agent samples candidate velocities on a polar grid, discards
those inside any neighbour's collision cone — widened to ``2.2 x
radius`` and translated by the neighbour's velocity, expressed as a pair
of half-planes — and takes the feasible sample closest to its desired
velocity toward the goal.

Every (agent, sample) pair is independent: the whole step is one
embarrassingly parallel evaluation, ``n_agents x n_samples`` threads on
the CUDA backend, one broadcast expression on the NumPy one.

One deliberate difference from pymapf: updates are *synchronous*. All
agents choose their velocity against the same snapshot of the world,
then move together. pymapf updates agents in registration order inside
a timestep, so earlier agents ignore later ones; the synchronous rule
is order-independent, which is what makes it parallel — and is also the
standard formulation of the decentralized problem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend, resolve_backend

__all__ = ["VelocityObstacleSim", "VOResult"]


@dataclass
class VOResult:
    """Trajectories and summary metrics of a velocity-obstacle run."""

    positions: np.ndarray  # (T+1, n_agents, 2)
    velocities: np.ndarray  # (T, n_agents, 2)
    runtime: float
    backend: str
    extra: dict = field(default_factory=dict)

    def goals_reached(self, goals: np.ndarray, tolerance: float) -> int:
        """Number of agents ending within ``tolerance`` of their goal."""
        final = self.positions[-1]
        return int(
            (np.linalg.norm(final - goals, axis=1) <= tolerance).sum()
        )

    def min_separation(self) -> float:
        """Smallest pairwise agent distance over the whole run."""
        n = self.positions.shape[1]
        if n < 2:
            return float("inf")
        best = np.inf
        for frame in self.positions:
            diff = frame[:, None, :] - frame[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            dist[np.arange(n), np.arange(n)] = np.inf
            best = min(best, float(dist.min()))
        return best


class VelocityObstacleSim:
    """Multi-agent velocity-obstacle simulation.

    Args:
        timestep: integration step in seconds.
        radius: agent radius; the collision cone uses ``2.2 x radius``,
            as in pymapf.
        vmax: maximum speed; also the desired cruise speed toward the
            goal.
        n_angles: angular resolution of the velocity sample grid.
        n_speeds: radial resolution of the velocity sample grid.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``.

    pymapf samples 20 angles x 5 speeds; the defaults match.
    """

    def __init__(
        self,
        timestep: float = 0.1,
        radius: float = 0.5,
        vmax: float = 2.0,
        n_angles: int = 20,
        n_speeds: int = 5,
        backend: Backend = "auto",
        profile: bool = False,
    ):
        self.timestep = float(timestep)
        self.radius = float(radius)
        self.vmax = float(vmax)
        self.backend = backend
        #: When True and the backend is CUDA, ``run`` reports per-phase
        #: wall times (``h2d``/``kernel``/``d2h``/``host``, seconds,
        #: summed over steps) in ``VOResult.extra``. Phase boundaries
        #: are device syncs, so profiled totals run slightly slower.
        self.profile = bool(profile)
        self._starts: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []
        self._obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        angles = np.linspace(0.0, 2.0 * np.pi, n_angles)
        speeds = np.linspace(0.0, self.vmax, n_speeds)
        vv, aa = np.meshgrid(speeds, angles)
        self._samples = np.stack(
            [(vv * np.cos(aa)).ravel(), (vv * np.sin(aa)).ravel()], axis=1
        )

    def add_agent(self, start, goal) -> None:
        """Register an agent by start and goal position (2D)."""
        self._starts.append(np.asarray(start, dtype=np.float64))
        self._goals.append(np.asarray(goal, dtype=np.float64))

    def add_obstacle(self, position, velocity) -> None:
        """Register a moving obstacle with constant velocity."""
        self._obstacles.append(
            (
                np.asarray(position, dtype=np.float64),
                np.asarray(velocity, dtype=np.float64),
            )
        )

    @property
    def goals(self) -> np.ndarray:
        """``(n_agents, 2)`` goal positions."""
        return np.stack(self._goals) if self._goals else np.empty((0, 2))

    def run(self, n_steps: int) -> VOResult:
        """Simulate ``n_steps`` timesteps and return the trajectories."""
        if not self._starts:
            raise ValueError("register at least one agent before running")
        started = time.perf_counter()
        which = resolve_backend(self.backend)
        n = len(self._starts)
        pos = np.stack(self._starts)
        vel = np.zeros((n, 2))
        goals = self.goals
        positions = [pos.copy()]
        velocities = []

        timings: dict[str, float] = {}
        if which == "cuda":
            step = self._make_cuda_step(
                timings if self.profile else None
            )
        else:
            step = self._step_cpu

        for k in range(n_steps):
            desired = self._desired_velocity(pos, goals)
            others = self._world_snapshot(pos, vel, k)
            vel = step(pos, vel, desired, others)
            pos = pos + vel * self.timestep
            positions.append(pos.copy())
            velocities.append(vel.copy())

        runtime = time.perf_counter() - started
        extra: dict = {}
        if timings:
            device = sum(timings.values())
            extra = dict(timings, host=max(runtime - device, 0.0))
        return VOResult(
            positions=np.stack(positions),
            velocities=np.stack(velocities),
            runtime=runtime,
            backend=which,
            extra=extra,
        )

    # -- shared pieces ---------------------------------------------------

    def _desired_velocity(self, pos: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Full speed toward the goal; zero within ``radius / 5`` of it."""
        disp = goals - pos
        norm = np.linalg.norm(disp, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(norm > 1e-12, disp / norm, 0.0)
        desired = self.vmax * unit
        desired[norm[:, 0] < self.radius / 5.0] = 0.0
        return desired

    def _world_snapshot(
        self, pos: np.ndarray, vel: np.ndarray, k: int
    ) -> np.ndarray:
        """``(n_agents + n_obstacles, 4)`` states every agent plans against."""
        rows = [np.concatenate([pos, vel], axis=1)]
        for p0, v in self._obstacles:
            rows.append(
                np.concatenate([p0 + v * k * self.timestep, v])[None, :]
            )
        return np.concatenate(rows, axis=0)

    # -- CPU reference ---------------------------------------------------

    def _step_cpu(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        desired: np.ndarray,
        others: np.ndarray,
    ) -> np.ndarray:
        """Vectorized half-plane feasibility over (agent, other, sample)."""
        n = len(pos)
        samples = self._samples  # (S, 2)
        margin = 2.2 * self.radius

        disp = pos[:, None, :] - others[None, :, :2]  # (n, K, 2)
        dist = np.maximum(np.linalg.norm(disp, axis=-1), margin)
        theta = np.arctan2(disp[..., 1], disp[..., 0])
        half = np.arcsin(np.clip(margin / dist, -1.0, 1.0))
        phi_l = theta + half  # (n, K)
        phi_r = theta - half

        rv = samples[None, :, :] - others[:, None, 2:]  # (K, S, 2)
        left = (
            np.sin(phi_l)[:, :, None] * rv[None, :, :, 0]
            - np.cos(phi_l)[:, :, None] * rv[None, :, :, 1]
        )  # (n, K, S)
        right = (
            np.sin(phi_r)[:, :, None] * rv[None, :, :, 0]
            - np.cos(phi_r)[:, :, None] * rv[None, :, :, 1]
        )
        inside = (left < 0.0) & (right > 0.0)
        inside[np.arange(n), np.arange(n), :] = False  # ignore self
        feasible = ~inside.any(axis=1)  # (n, S)

        objective = np.linalg.norm(
            samples[None, :, :] - desired[:, None, :], axis=-1
        )
        objective[~feasible] = np.inf
        best = np.argmin(objective, axis=1)
        chosen = samples[best]
        chosen[~np.isfinite(objective[np.arange(n), best])] = 0.0
        return chosen

    # -- CUDA backend ----------------------------------------------------

    def _make_cuda_step(self, timings: dict | None = None):
        import cupy

        from .kernels import get_kernel

        kernel = get_kernel("velocity_obstacles", "score_samples")
        samples_d = cupy.asarray(self._samples)
        n_samples = len(self._samples)
        radius = self.radius
        samples_h = self._samples
        sync = cupy.cuda.get_current_stream().synchronize

        def tick(phase, mark):
            sync()
            now = time.perf_counter()
            timings[phase] = timings.get(phase, 0.0) + now - mark
            return now

        def step(pos, vel, desired, others):
            n = len(pos)
            if timings is not None:
                sync()
                mark = time.perf_counter()
            states = cupy.asarray(np.concatenate([pos, vel], axis=1))
            others_d = cupy.asarray(others)
            desired_d = cupy.asarray(desired)
            self_index = cupy.arange(n, dtype=cupy.int32)
            scores = cupy.empty((n, n_samples), dtype=cupy.float64)
            if timings is not None:
                mark = tick("h2d", mark)
            threads = 256
            blocks = min(
                65535, (n * n_samples + threads - 1) // threads
            )
            kernel(
                (blocks,),
                (threads,),
                (
                    states,
                    desired_d,
                    others_d,
                    self_index,
                    samples_d,
                    scores,
                    np.int32(n),
                    np.int32(len(others)),
                    np.int32(n_samples),
                    np.float64(radius),
                ),
            )
            best_d = cupy.argmin(scores, axis=1)
            mins_d = scores.min(axis=1)
            if timings is not None:
                mark = tick("kernel", mark)
            best = cupy.asnumpy(best_d)
            mins = cupy.asnumpy(mins_d)
            if timings is not None:
                tick("d2h", mark)
            chosen = samples_h[best]
            chosen[~np.isfinite(mins)] = 0.0
            return chosen

        return step
