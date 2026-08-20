# Velocity obstacles

`cuplan.VelocityObstacleSim` — decentralized collision avoidance with
velocity obstacles (Fiorini and Shiller 1998, *Motion planning in
dynamic environments using velocity obstacles*, IJRR 17(7)), mirroring
`pymapf.decentralized.velocity_obstacle`.

## Semantics

Each timestep, each agent:

1. computes a desired velocity — full speed toward its goal, zero once
   within `radius / 5` of it;
2. builds, for every other agent and moving obstacle, a collision cone
   widened to `2.2 x radius` and translated by that obstacle's
   velocity, expressed as two half-planes;
3. samples candidate velocities on a polar grid (20 angles x 5 speeds
   by default, as in pymapf), discards samples inside any cone, and
   takes the feasible sample closest to the desired velocity — or
   stops when nothing is feasible.

## One deliberate difference from pymapf

Updates are **synchronous**: all agents choose against the same
snapshot of the world, then move together. pymapf updates agents in
registration order within a timestep — earlier agents do not see later
ones at all. The synchronous rule is order-independent, which is what
makes it parallel, and is the standard formulation of the decentralized
problem. Because the two simulators integrate different dynamics,
the benchmark reports cuplan CPU vs CUDA only for this family rather
than a misleading cross-library wall-clock number.

## Parallelization

Every (agent, sample) pair is independent — the step is one
embarrassingly parallel evaluation:

- **CPU reference** — one broadcast expression over
  `(agents, others, samples)`.
- **CUDA** — one thread per (agent, sample), looping over the others;
  the per-agent argmin is a device-side reduction.

CPU and CUDA trajectories agree to floating-point tolerance (tested).

::: cuplan.velocity_obstacles
