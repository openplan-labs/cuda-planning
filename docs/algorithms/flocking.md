# Flocking

`cuplan.FlockingSim` — Reynolds' Boids (Reynolds 1987, *Flocks, herds
and schools: a distributed behavioral model*, SIGGRAPH), mirroring the
`boids` behavior of `pymapf.swarm.flocking`.

## Semantics

Within a perception radius, each agent accumulates three steering
accelerations:

- **separation** — inverse-square repulsion from neighbours closer
  than the separation distance;
- **cohesion** — toward the mean neighbour offset;
- **alignment** — toward the mean neighbour velocity.

The command is clamped to a maximum acceleration, integrated with a
capped speed. Works in 2D and 3D. `FlockingResult` reports Vicsek's
polarization order parameter and a mean-neighbour-distance cohesion
proxy, so "does it flock?" is a number rather than an impression.

## Parallelization

The per-agent force is a gather over neighbours with no dependencies
between agents:

- **CPU reference** — full pairwise NumPy broadcast.
- **CUDA** — one thread per agent scanning the swarm.

The O(n²) neighbour scan is deliberate on both backends: at the swarm
sizes this library targets (up to a few thousand agents), rebuilding a
spatial index every step costs more than it saves, and the two
backends stay exactly comparable — CPU and CUDA trajectories agree to
floating-point tolerance (tested).

::: cuplan.flocking
