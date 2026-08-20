# PIBT

`cuplan.PIBT` — Priority Inheritance with Backtracking (Okumura,
Machida, Défago and Tamura 2022, *Artificial Intelligence*
310:103752). No whole-path planning: every timestep, each agent
proposes the neighbouring vertex closest to its goal, and conflicts are
settled on the spot — a high-priority agent lends its priority to the
occupant of the vertex it wants, recursively, backtracking when a chain
cannot move. Priorities grow while an agent is away from its goal and
reset on arrival, which prevents starvation.

## Guarantees, stated plainly

PIBT is **incomplete**: it can livelock on instances that require an
agent to move far away from its goal. Reachability of every goal is
verified up front, so a failure is reported as livelock within the
step bound, never silently. LaCAM — [on the roadmap](../roadmap.md) —
wraps PIBT in a complete search and fixes exactly this.

## Parallelization

Three parts, two of them parallel:

1. **Distance oracle** — one exact goal-distance table per agent, the
   dominant cost in pymapf's implementation (a serial Dijkstra per
   agent). Here it is a single [batched BFS](bfs.md).
2. **Candidate evaluation** — each step, every agent's five candidate
   vertices are gathered and ordered by goal distance with random
   tie-breaking: one vectorized gather + argsort across all agents.
3. **Inheritance chains** — recursive and data-dependent, so they run
   on the host, exactly as written in the paper.

Runs are reproducible for a fixed `seed`, and identical between the
CPU and CUDA backends (the backends change where the oracle is
computed, not any decision).

::: cuplan.pibt
