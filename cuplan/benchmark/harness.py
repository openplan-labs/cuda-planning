"""Benchmark runners: same scenarios, every solver, honest numbers.

Each record carries the machine-independent facts (solver, backend,
grid size, agent count, seed) and the measured ones (wall time, sum of
costs, success). Wall time covers the full ``solve`` call including
host/device transfers — the number a user would actually see.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from ..backend import cuda_available
from ..pibt import PIBT
from ..prioritized import PrioritizedPlanning
from .scenarios import Scenario, random_scenario

__all__ = ["BenchmarkResult", "run_mapf_benchmark", "run_vo_benchmark"]


@dataclass
class BenchmarkResult:
    """One (scenario, solver) measurement."""

    family: str  # "prioritized" | "pibt" | "bfs" | "velocity_obstacles"
    solver: str  # e.g. "cuplan-cuda", "pymapf"
    size: int
    n_agents: int
    seed: int
    success: bool
    runtime: float  # seconds, full solve() including transfers
    cost: int | None = None  # sum of costs, when solved
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def machine_description() -> str:
    """One-line description of the benchmark machine."""
    gpu = ""
    if cuda_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        gpu = f", GPU {props['name'].decode()}"
    return f"{platform.processor() or platform.machine()}{gpu}"


def _time_solver(solve: Callable[[], object]) -> tuple:
    started = time.perf_counter()
    solution = solve()
    return solution, time.perf_counter() - started


def run_mapf_benchmark(
    sizes: list[int],
    agent_counts: list[int],
    seeds: list[int],
    obstacle_density: float = 0.15,
    include_pymapf: bool = True,
    include_cuda: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[BenchmarkResult]:
    """Run prioritized planning and PIBT across scenario axes.

    Every (size, agents, seed) triple builds one scenario handed to all
    solvers. Agent counts that do not fit a grid size are skipped.

    Returns the flat list of records; aggregation is the reporter's job.
    """
    include_cuda = cuda_available() if include_cuda is None else include_cuda
    say = progress or (lambda s: None)
    results: list[BenchmarkResult] = []

    for size in sizes:
        for n_agents in agent_counts:
            if 2 * n_agents > size * size * (1 - obstacle_density) * 0.5:
                continue
            for seed in seeds:
                scenario = random_scenario(
                    size, n_agents, obstacle_density, seed
                )
                say(f"{size}x{size}, {n_agents} agents, seed {seed}")
                results.extend(
                    _run_mapf_instance(
                        scenario, include_pymapf, include_cuda
                    )
                )
    return results


def _run_mapf_instance(
    scenario: Scenario, include_pymapf: bool, include_cuda: bool
) -> list[BenchmarkResult]:
    records: list[BenchmarkResult] = []
    problem = scenario.to_cuplan()
    common = dict(
        size=scenario.grid.height,
        n_agents=scenario.n_agents,
        seed=scenario.seed,
    )

    backends = ["cpu"] + (["cuda"] if include_cuda else [])
    for backend in backends:
        for family, solver in (
            ("prioritized", PrioritizedPlanning(backend=backend)),
            ("pibt", PIBT(backend=backend)),
        ):
            solution, runtime = _time_solver(lambda s=solver: s.solve(problem))
            ok = solution is not None and solution.is_valid()
            records.append(
                BenchmarkResult(
                    family=family,
                    solver=f"cuplan-{backend}",
                    success=ok,
                    runtime=runtime,
                    cost=solution.sum_of_costs if ok else None,
                    **common,
                )
            )

    if include_pymapf:
        try:
            import pymapf.algorithms  # noqa: F401 - registers solvers
            from pymapf.core.solver import get_solver
        except ImportError:
            return records
        pproblem = scenario.to_pymapf()
        for family in ("prioritized", "pibt"):
            solver = get_solver(family)
            solution, runtime = _time_solver(
                lambda s=solver: s.solve(pproblem)
            )
            ok = solution is not None and solution.is_valid()
            records.append(
                BenchmarkResult(
                    family=family,
                    solver="pymapf",
                    success=ok,
                    runtime=runtime,
                    cost=solution.sum_of_costs if ok else None,
                    **common,
                )
            )
    return records


def run_bfs_benchmark(
    sizes: list[int],
    batch_sizes: list[int],
    seeds: list[int],
    obstacle_density: float = 0.15,
    include_cuda: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[BenchmarkResult]:
    """Benchmark the batched distance-map primitive: CPU vs CUDA.

    This is the primitive every solver consumes (heuristic tables, the
    PIBT oracle), measured directly: one flood fill per source, batched.
    pymapf has no batched equivalent — its per-goal Dijkstra cost is
    included in the solver families' timings.
    """
    from ..bfs import distance_maps

    include_cuda = cuda_available() if include_cuda is None else include_cuda
    say = progress or (lambda s: None)
    results: list[BenchmarkResult] = []
    backends = ["cpu"] + (["cuda"] if include_cuda else [])

    for size in sizes:
        for batch in batch_sizes:
            for seed in seeds:
                scenario = random_scenario(
                    size,
                    min(batch, int(size * size * 0.2)),
                    obstacle_density,
                    seed,
                )
                goals = np.asarray(scenario.goals)
                for backend in backends:
                    say(f"bfs {size}x{size} batch {len(goals)} {backend}")
                    if backend == "cuda":  # warm the NVRTC cache
                        distance_maps(scenario.grid, goals[:1], backend="cuda")
                    started = time.perf_counter()
                    distance_maps(scenario.grid, goals, backend=backend)
                    runtime = time.perf_counter() - started
                    results.append(
                        BenchmarkResult(
                            family="bfs",
                            solver=f"cuplan-{backend}",
                            size=size,
                            n_agents=len(goals),
                            seed=seed,
                            success=True,
                            runtime=runtime,
                        )
                    )
    return results


def run_vo_benchmark(
    agent_counts: list[int],
    seeds: list[int],
    n_steps: int = 80,
    include_cuda: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[BenchmarkResult]:
    """Benchmark velocity-obstacle steps: cuplan CPU vs CUDA.

    Agents start on a circle with antipodal goals — the classic
    all-cross stress case. pymapf's simulator is not timed here: its
    sequential in-step update solves a different problem per agent (see
    :mod:`cuplan.velocity_obstacles`), so wall-clock comparison would
    be misleading; the MAPF families carry the cross-library numbers.
    """
    from ..velocity_obstacles import VelocityObstacleSim

    include_cuda = cuda_available() if include_cuda is None else include_cuda
    say = progress or (lambda s: None)
    results: list[BenchmarkResult] = []
    backends = ["cpu"] + (["cuda"] if include_cuda else [])

    for n_agents in agent_counts:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            angles = np.sort(rng.uniform(0, 2 * np.pi, n_agents))
            r = 2.0 + 0.35 * n_agents
            starts = r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
            goals = -starts
            for backend in backends:
                say(f"vo {n_agents} agents seed {seed} {backend}")
                sim = VelocityObstacleSim(backend=backend)
                for s, g in zip(starts, goals, strict=True):
                    sim.add_agent(s, g)
                run = sim.run(n_steps)
                reached = run.goals_reached(goals, tolerance=0.5)
                results.append(
                    BenchmarkResult(
                        family="velocity_obstacles",
                        solver=f"cuplan-{backend}",
                        size=0,
                        n_agents=n_agents,
                        seed=seed,
                        success=True,
                        runtime=run.runtime,
                        cost=None,
                        extra={
                            "goals_reached": float(reached),
                            "min_separation": run.min_separation(),
                        },
                    )
                )
    return results
