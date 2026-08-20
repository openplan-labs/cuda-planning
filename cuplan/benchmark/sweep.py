"""Checkpointed benchmark sweep behind the Experiments pages.

This is the long-form companion to :mod:`cuplan.benchmark.harness`:
agents x grid x density scaling for every solver family, solution
quality on identical seeded instances, per-phase CUDA timing, and
throughput curves — written to CSV *as each cell finishes*, so a
killed sweep resumes where it stopped instead of starting over.

Every record carries its full conditions (family, solver, grid size,
agent count, obstacle density, seed) and an explicit ``status``:

``solved``
    the solver returned a valid solution.
``unsolved``
    the solver terminated and reported failure (incomplete solvers
    failing on hard instances — a result, not an error).
``timeout``
    the solver was stopped at the per-instance wall-clock cap; the
    recorded runtime is the cap. Never extrapolated.
``skipped``
    the cell was not run, with the reason in ``extra`` — either every
    seed of a smaller cell already timed out for this solver, or the
    cell is outside a documented resource cap.
``error``
    the solver raised (out-of-memory included); the reason is in
    ``extra``.

Run stages with ``python -m cuplan.benchmark.sweep --stage <name>``;
see ``--help`` for the axes. Timeouts use ``SIGALRM``, so the sweep is
Unix-only (the library itself is not).
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..backend import cuda_available
from ..bfs import distance_maps
from ..pibt import PIBT
from ..prioritized import PrioritizedPlanning
from .scenarios import Scenario, random_scenario

__all__ = [
    "SweepRecord",
    "Checkpoint",
    "run_mapf_sweep",
    "run_bfs_sweep",
    "run_vo_sweep",
    "run_flocking_sweep",
    "run_phase_sweep",
]

FIELDS = [
    "family",
    "solver",
    "size",
    "n_agents",
    "density",
    "seed",
    "status",
    "runtime",
    "cost",
    "makespan",
    "extra",
]


@dataclass
class SweepRecord:
    """One (instance, solver) cell of the sweep, conditions included."""

    family: str
    solver: str  # "pymapf" | "cuplan-cpu" | "cuplan-cuda"
    size: int  # grid side; 0 for continuous-space families
    n_agents: int  # agents, or batch size for the BFS primitive
    density: float  # obstacle density; 0.0 for continuous space
    seed: int
    status: str  # solved | unsolved | timeout | skipped | error
    runtime: float | None = None  # seconds; the cap itself for timeouts
    cost: int | None = None  # sum of costs, when solved
    makespan: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (
            self.family,
            self.solver,
            self.size,
            self.n_agents,
            round(self.density, 4),
            self.seed,
        )

    def as_row(self) -> dict:
        row = asdict(self)
        row["extra"] = json.dumps(self.extra, sort_keys=True)
        return row


def read_records(path: Path) -> list[SweepRecord]:
    """Load a sweep CSV back into records (for reports and figures)."""
    records = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            records.append(
                SweepRecord(
                    family=row["family"],
                    solver=row["solver"],
                    size=int(row["size"]),
                    n_agents=int(row["n_agents"]),
                    density=float(row["density"]),
                    seed=int(row["seed"]),
                    status=row["status"],
                    runtime=float(row["runtime"]) if row["runtime"] else None,
                    cost=int(float(row["cost"])) if row["cost"] else None,
                    makespan=(
                        int(float(row["makespan"])) if row["makespan"] else None
                    ),
                    extra=json.loads(row["extra"]) if row["extra"] else {},
                )
            )
    return records


class Checkpoint:
    """Append-per-record CSV so partial progress survives a kill.

    Existing rows are loaded on open and their cells are skipped on
    re-run; delete rows (or the file) to re-measure them.
    """

    def __init__(self, path: Path):
        self.path = path
        self.done: set[tuple] = set()
        if path.exists():
            self.done = {r.key for r in read_records(path)}
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    def has(self, record_key: tuple) -> bool:
        return record_key in self.done

    def append(self, record: SweepRecord) -> None:
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(
                record.as_row()
            )
        self.done.add(record.key)


class _Timeout(Exception):
    pass


def _call_with_timeout(fn: Callable[[], object], seconds: float):
    """Run ``fn`` under a SIGALRM deadline; returns (status, value, dt).

    ``status`` is ``"ok"`` or ``"timeout"``; on timeout ``dt`` is the
    cap. Main-thread only, Unix only — exactly the sweep's situation.
    """

    def _raise(signum, frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    try:
        value = fn()
        return "ok", value, time.perf_counter() - started
    except _Timeout:
        return "timeout", None, seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _free_device_memory() -> None:
    if cuda_available():
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()


def _error_record(base: SweepRecord, exc: Exception) -> SweepRecord:
    reason = f"{type(exc).__name__}: {exc}"[:200]
    base.status = "error"
    base.extra = {"reason": reason}
    return base


# ---------------------------------------------------------------------------
# MAPF families: prioritized planning and PIBT, three solvers
# ---------------------------------------------------------------------------


def _solve_cuplan(family: str, backend: str, problem):
    solver = (
        PrioritizedPlanning(backend=backend)
        if family == "prioritized"
        else PIBT(backend=backend)
    )
    return solver.solve(problem)


def _solve_pymapf(family: str, scenario: Scenario):
    import pymapf.algorithms  # noqa: F401 - registers solvers
    from pymapf.core.solver import get_solver

    return get_solver(family).solve(scenario.to_pymapf())


def run_mapf_sweep(
    out: Path,
    sizes: list[int],
    agent_counts: list[int],
    densities: list[float],
    seeds: list[int],
    pymapf_timeout: float = 60.0,
    cuplan_timeout: float = 300.0,
    pymapf_max_size: int = 128,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Scaling sweep over agents x grid x density for both MAPF families.

    Identical seeded instances go to every solver. pymapf runs are
    capped at ``pymapf_timeout`` seconds and skipped above
    ``pymapf_max_size`` grids; once every seed of a cell times out for
    a solver, larger agent counts on the same (family, size, density)
    axis are recorded as skipped rather than burning the cap again.
    """
    say = progress or (lambda s: None)
    ckpt = Checkpoint(out / "mapf.csv")
    has_cuda = cuda_available()
    # (family, solver, size, density) -> smallest all-seeds-timed-out n
    dead: dict[tuple, int] = {}

    solvers = ["cuplan-cpu"] + (["cuplan-cuda"] if has_cuda else [])
    for size in sorted(sizes):
        for density in densities:
            for n_agents in sorted(agent_counts):
                if 2 * n_agents > size * size * (1 - density) * 0.5:
                    continue
                cell_records: dict[str, list[SweepRecord]] = {}
                for seed in seeds:
                    scenario = None
                    for family in ("prioritized", "pibt"):
                        runs = [(s, None) for s in solvers]
                        if size <= pymapf_max_size:
                            runs.append(("pymapf", None))
                        for solver_name, _ in runs:
                            record = SweepRecord(
                                family=family,
                                solver=solver_name,
                                size=size,
                                n_agents=n_agents,
                                density=density,
                                seed=seed,
                                status="skipped",
                            )
                            if ckpt.has(record.key):
                                continue
                            axis = (family, solver_name, size, density)
                            if axis in dead and n_agents > dead[axis]:
                                record.extra = {
                                    "reason": (
                                        "all seeds timed out at "
                                        f"{dead[axis]} agents"
                                    )
                                }
                                ckpt.append(record)
                                continue
                            if scenario is None:
                                scenario = random_scenario(
                                    size, n_agents, density, seed
                                )
                            say(
                                f"mapf {family} {solver_name} {size}x{size} "
                                f"d={density:.0%} n={n_agents} seed={seed}"
                            )
                            record = _run_mapf_cell(
                                record,
                                scenario,
                                family,
                                solver_name,
                                pymapf_timeout
                                if solver_name == "pymapf"
                                else cuplan_timeout,
                            )
                            ckpt.append(record)
                            cell_records.setdefault(
                                (family, solver_name), []
                            ).append(record)
                            _free_device_memory()
                # Escalation: a cell whose every seed timed out kills
                # the rest of its (family, solver, size, density) axis.
                for (family, solver_name), rows in cell_records.items():
                    if rows and all(r.status == "timeout" for r in rows):
                        axis = (family, solver_name, size, density)
                        dead.setdefault(axis, n_agents)
    return ckpt.path


def _run_mapf_cell(
    record: SweepRecord,
    scenario: Scenario,
    family: str,
    solver_name: str,
    timeout: float,
) -> SweepRecord:
    try:
        if solver_name == "pymapf":
            fn = lambda: _solve_pymapf(family, scenario)  # noqa: E731
        else:
            backend = solver_name.split("-", 1)[1]
            problem = scenario.to_cuplan()
            fn = lambda: _solve_cuplan(family, backend, problem)  # noqa: E731
        status, solution, runtime = _call_with_timeout(fn, timeout)
    except Exception as exc:  # OOM and friends: report, keep sweeping
        return _error_record(record, exc)
    record.runtime = runtime
    if status == "timeout":
        record.status = "timeout"
        record.extra = {"timeout_s": timeout}
        return record
    ok = solution is not None and solution.is_valid()
    record.status = "solved" if ok else "unsolved"
    if ok:
        record.cost = int(solution.sum_of_costs)
        record.makespan = int(solution.makespan)
    return record


# ---------------------------------------------------------------------------
# Batched BFS primitive: throughput vs batch size
# ---------------------------------------------------------------------------


def run_bfs_sweep(
    out: Path,
    sizes: list[int],
    batch_sizes: list[int],
    seeds: list[int],
    density: float = 0.15,
    cpu_max_batch: int = 1024,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Batched distance-map builds, CPU vs CUDA, across batch sizes.

    CPU runs above ``cpu_max_batch`` sources are recorded as skipped
    (a 2048-map build already costs ~2 minutes per seed on this CPU);
    the CUDA side runs the full axis.
    """
    say = progress or (lambda s: None)
    ckpt = Checkpoint(out / "bfs.csv")
    backends = ["cpu"] + (["cuda"] if cuda_available() else [])

    for size in sorted(sizes):
        for batch in sorted(batch_sizes):
            for seed in seeds:
                scenario = None
                for backend in backends:
                    record = SweepRecord(
                        family="bfs",
                        solver=f"cuplan-{backend}",
                        size=size,
                        n_agents=batch,
                        density=density,
                        seed=seed,
                        status="skipped",
                    )
                    if ckpt.has(record.key):
                        continue
                    if backend == "cpu" and batch > cpu_max_batch:
                        record.extra = {
                            "reason": f"cpu capped at {cpu_max_batch} sources"
                        }
                        ckpt.append(record)
                        continue
                    if scenario is None:
                        scenario = random_scenario(
                            size,
                            min(batch, int(size * size * 0.2)),
                            density,
                            seed,
                        )
                    # Sources beyond the agent cap: sample free cells.
                    rng = np.random.default_rng(seed)
                    free = np.argwhere(
                        distance_maps(
                            scenario.grid,
                            np.asarray(scenario.starts[:1]),
                            backend="cpu",
                        )[0]
                        >= 0
                    )
                    picks = rng.choice(
                        len(free), size=min(batch, len(free)), replace=False
                    )
                    sources = free[picks].astype(np.int32)
                    say(f"bfs {size}x{size} batch {len(sources)} {backend}")
                    try:
                        if backend == "cuda":  # warm NVRTC + context
                            distance_maps(
                                scenario.grid, sources[:1], backend="cuda"
                            )
                        started = time.perf_counter()
                        distance_maps(scenario.grid, sources, backend=backend)
                        record.runtime = time.perf_counter() - started
                        record.status = "solved"
                        record.n_agents = len(sources)
                        record.extra = {
                            "maps_per_s": len(sources) / record.runtime
                        }
                    except Exception as exc:
                        record = _error_record(record, exc)
                    ckpt.append(record)
                    _free_device_memory()
    return ckpt.path


# ---------------------------------------------------------------------------
# Velocity obstacles and flocking: steps/sec vs swarm size
# ---------------------------------------------------------------------------


def _vo_sim(n_agents: int, seed: int, backend: str, profile: bool = False):
    from ..velocity_obstacles import VelocityObstacleSim

    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_agents))
    r = 2.0 + 0.35 * n_agents
    starts = r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    sim = VelocityObstacleSim(backend=backend, profile=profile)
    for s, g in zip(starts, -starts, strict=True):
        sim.add_agent(s, g)
    return sim, -starts


def run_vo_sweep(
    out: Path,
    agent_counts: list[int],
    seeds: list[int],
    n_steps: int = 80,
    cpu_max_agents: int = 1024,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Velocity-obstacle runs (antipodal circle, 80 steps) vs agents."""
    say = progress or (lambda s: None)
    ckpt = Checkpoint(out / "vo.csv")
    backends = ["cpu"] + (["cuda"] if cuda_available() else [])

    for n_agents in sorted(agent_counts):
        for seed in seeds:
            for backend in backends:
                record = SweepRecord(
                    family="velocity_obstacles",
                    solver=f"cuplan-{backend}",
                    size=0,
                    n_agents=n_agents,
                    density=0.0,
                    seed=seed,
                    status="skipped",
                )
                if ckpt.has(record.key):
                    continue
                if backend == "cpu" and n_agents > cpu_max_agents:
                    record.extra = {
                        "reason": (
                            f"cpu capped at {cpu_max_agents} agents "
                            "(O(n^2 x samples) temporaries exceed host RAM)"
                        )
                    }
                    ckpt.append(record)
                    continue
                say(f"vo {n_agents} agents seed {seed} {backend}")
                try:
                    sim, goals = _vo_sim(n_agents, seed, backend)
                    if backend == "cuda":  # warm NVRTC + context
                        warm, _ = _vo_sim(min(n_agents, 8), seed, backend)
                        warm.run(1)
                    run = sim.run(n_steps)
                    record.status = "solved"
                    record.runtime = run.runtime
                    record.extra = {
                        "steps": n_steps,
                        "agent_steps_per_s": n_agents * n_steps / run.runtime,
                        "goals_reached": float(
                            run.goals_reached(goals, tolerance=0.5)
                        ),
                        "min_separation": run.min_separation(),
                    }
                except Exception as exc:
                    record = _error_record(record, exc)
                ckpt.append(record)
                _free_device_memory()
    return ckpt.path


def run_flocking_sweep(
    out: Path,
    agent_counts: list[int],
    seeds: list[int],
    n_steps: int = 200,
    cpu_max_agents: int = 2048,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Boids runs (uniform cube start, 200 steps) vs swarm size."""
    from ..flocking import FlockingSim

    say = progress or (lambda s: None)
    ckpt = Checkpoint(out / "flocking.csv")
    backends = ["cpu"] + (["cuda"] if cuda_available() else [])

    for n_agents in sorted(agent_counts):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            side = 2.0 * np.sqrt(n_agents)
            positions = rng.uniform(-side, side, (n_agents, 2))
            velocities = rng.uniform(-1.0, 1.0, (n_agents, 2))
            for backend in backends:
                record = SweepRecord(
                    family="flocking",
                    solver=f"cuplan-{backend}",
                    size=0,
                    n_agents=n_agents,
                    density=0.0,
                    seed=seed,
                    status="skipped",
                )
                if ckpt.has(record.key):
                    continue
                if backend == "cpu" and n_agents > cpu_max_agents:
                    record.extra = {
                        "reason": (
                            f"cpu capped at {cpu_max_agents} agents "
                            "(O(n^2) pairwise temporaries)"
                        )
                    }
                    ckpt.append(record)
                    continue
                say(f"flocking {n_agents} agents seed {seed} {backend}")
                try:
                    sim = FlockingSim(
                        positions, velocities, backend=backend
                    )
                    if backend == "cuda":  # warm NVRTC + context
                        FlockingSim(
                            positions[:8], velocities[:8], backend="cuda"
                        ).run(1)
                    run = sim.run(n_steps)
                    record.status = "solved"
                    record.runtime = run.runtime
                    record.extra = {
                        "steps": n_steps,
                        "agent_steps_per_s": n_agents * n_steps / run.runtime,
                        "final_polarization": float(run.polarization()[-1]),
                    }
                except Exception as exc:
                    record = _error_record(record, exc)
                ckpt.append(record)
                _free_device_memory()
    return ckpt.path


# ---------------------------------------------------------------------------
# CUDA phase breakdown: H2D / kernel / D2H / host
# ---------------------------------------------------------------------------

PHASE_FIELDS = [
    "family",
    "size",
    "n",
    "density",
    "seed",
    "phase",
    "seconds",
]


def run_phase_sweep(
    out: Path,
    seeds: list[int],
    bfs_size: int = 256,
    bfs_batches: list[int] | None = None,
    vo_agents: list[int] | None = None,
    flocking_agents: list[int] | None = None,
    prioritized_cells: list[tuple[int, int]] | None = None,
    density: float = 0.15,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Time the CUDA phases of each family; long-format CSV.

    Primitive families report H2D / kernel / D2H / host from device
    syncs inserted at phase boundaries (so a profiled run is slightly
    slower than an unprofiled one — the split explains the crossover,
    the unprofiled sweeps carry the headline totals). Prioritized
    planning reports a solver-level split instead: oracle (batched
    BFS), search (space-time wavefronts) and reserve (table updates).
    """
    if not cuda_available():
        raise RuntimeError("phase sweep needs a working CUDA device")
    import cupy  # noqa: F401

    from ..astar import space_time_astar
    from ..flocking import FlockingSim
    from ..reservations import ReservationTable

    say = progress or (lambda s: None)
    bfs_batches = bfs_batches or [16, 64, 256, 1024]
    vo_agents = vo_agents or [16, 64, 256, 1024]
    flocking_agents = flocking_agents or [256, 1024, 4096]
    prioritized_cells = prioritized_cells or [(64, 64), (64, 256), (128, 256)]

    path = out / "phases.csv"
    done: set[tuple] = set()
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                done.add(
                    (
                        row["family"],
                        int(row["size"]),
                        int(row["n"]),
                        int(row["seed"]),
                    )
                )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=PHASE_FIELDS).writeheader()

    def emit(family, size, n, seed, phases: dict[str, float]):
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PHASE_FIELDS)
            for phase, seconds in phases.items():
                writer.writerow(
                    {
                        "family": family,
                        "size": size,
                        "n": n,
                        "density": density if size else 0.0,
                        "seed": seed,
                        "phase": phase,
                        "seconds": seconds,
                    }
                )
        done.add((family, size, n, seed))

    for batch in bfs_batches:
        for seed in seeds:
            if ("bfs", bfs_size, batch, seed) in done:
                continue
            say(f"phases bfs {bfs_size}x{bfs_size} batch {batch} seed {seed}")
            scenario = random_scenario(
                bfs_size, min(batch, bfs_size * bfs_size // 5), density, seed
            )
            rng = np.random.default_rng(seed)
            free = np.argwhere(scenario.grid.free)
            sources = free[
                rng.choice(len(free), size=batch, replace=False)
            ].astype(np.int32)
            # Sources may sit in minor components; that only shortens
            # some maps and every backend sees the same sources.
            distance_maps(scenario.grid, sources[:1], backend="cuda")  # warm
            timings: dict[str, float] = {}
            started = time.perf_counter()
            distance_maps(
                scenario.grid, sources, backend="cuda", timings=timings
            )
            total = time.perf_counter() - started
            timings["host"] = max(total - sum(timings.values()), 0.0)
            emit("bfs", bfs_size, batch, seed, timings)
            _free_device_memory()

    for n_agents in vo_agents:
        for seed in seeds:
            if ("velocity_obstacles", 0, n_agents, seed) in done:
                continue
            say(f"phases vo {n_agents} agents seed {seed}")
            warm, _ = _vo_sim(8, seed, "cuda")
            warm.run(1)
            sim, _goals = _vo_sim(n_agents, seed, "cuda", profile=True)
            run = sim.run(80)
            emit("velocity_obstacles", 0, n_agents, seed, run.extra)
            _free_device_memory()

    for n_agents in flocking_agents:
        for seed in seeds:
            if ("flocking", 0, n_agents, seed) in done:
                continue
            say(f"phases flocking {n_agents} agents seed {seed}")
            rng = np.random.default_rng(seed)
            side = 2.0 * np.sqrt(n_agents)
            positions = rng.uniform(-side, side, (n_agents, 2))
            velocities = rng.uniform(-1.0, 1.0, (n_agents, 2))
            FlockingSim(positions[:8], velocities[:8], backend="cuda").run(1)
            sim = FlockingSim(
                positions, velocities, backend="cuda", profile=True
            )
            run = sim.run(200)
            emit("flocking", 0, n_agents, seed, run.extra)
            _free_device_memory()

    for size, n_agents in prioritized_cells:
        for seed in seeds:
            if ("prioritized", size, n_agents, seed) in done:
                continue
            say(
                f"phases prioritized {size}x{size} {n_agents} agents "
                f"seed {seed}"
            )
            scenario = random_scenario(size, n_agents, density, seed)
            problem = scenario.to_cuplan()
            distance_maps(problem.grid, problem.goals[:1], backend="cuda")
            phases = {"oracle": 0.0, "search": 0.0, "reserve": 0.0}
            started = time.perf_counter()
            mark = started
            dists = distance_maps(
                problem.grid, problem.goals, backend="cuda"
            )
            phases["oracle"] = time.perf_counter() - mark
            starts = problem.starts
            goal_dist = dists[
                np.arange(len(problem.agents)), starts[:, 0], starts[:, 1]
            ]
            horizon = int(2 * goal_dist.max() + 2 * len(problem.agents) + 16)
            table = ReservationTable(problem.grid, horizon, xp=cupy)
            for agent in problem.agents:
                mark = time.perf_counter()
                p = space_time_astar(
                    problem.grid,
                    agent.start,
                    agent.goal,
                    table=table,
                    horizon=horizon,
                    backend="cuda",
                )
                phases["search"] += time.perf_counter() - mark
                if p is None:
                    break
                mark = time.perf_counter()
                table.reserve_path(p)
                phases["reserve"] += time.perf_counter() - mark
            phases["host"] = max(
                time.perf_counter() - started - sum(phases.values()), 0.0
            )
            emit("prioritized", size, n_agents, seed, phases)
            _free_device_memory()
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuplan.benchmark.sweep",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--stage",
        choices=["mapf", "bfs", "vo", "flocking", "phases", "all"],
        default="all",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/experiments")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[32, 64, 128, 256]
    )
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--densities", type=float, nargs="+", default=[0.05, 0.15, 0.25]
    )
    parser.add_argument("--pymapf-timeout", type=float, default=60.0)
    parser.add_argument("--cuplan-timeout", type=float, default=300.0)
    parser.add_argument("--pymapf-max-size", type=int, default=128)
    parser.add_argument(
        "--bfs-sizes", type=int, nargs="+", default=[64, 128, 256]
    )
    parser.add_argument(
        "--bfs-batches", type=int, nargs="+", default=[16, 64, 256, 1024, 2048]
    )
    parser.add_argument(
        "--vo-agents",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--flocking-agents",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 2048, 4096, 8192],
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    say = lambda s: print(f"  {s}", flush=True)  # noqa: E731

    print(f"CUDA available: {cuda_available()}", flush=True)
    if args.stage in ("bfs", "all"):
        print("Stage: batched BFS", flush=True)
        run_bfs_sweep(
            args.out, args.bfs_sizes, args.bfs_batches, args.seeds,
            progress=say,
        )
    if args.stage in ("vo", "all"):
        print("Stage: velocity obstacles", flush=True)
        run_vo_sweep(args.out, args.vo_agents, args.seeds, progress=say)
    if args.stage in ("flocking", "all"):
        print("Stage: flocking", flush=True)
        run_flocking_sweep(
            args.out, args.flocking_agents, args.seeds, progress=say
        )
    if args.stage in ("phases", "all"):
        print("Stage: CUDA phase breakdown", flush=True)
        run_phase_sweep(args.out, args.seeds, progress=say)
    if args.stage in ("mapf", "all"):
        print("Stage: MAPF families", flush=True)
        run_mapf_sweep(
            args.out,
            args.sizes,
            args.agents,
            args.densities,
            args.seeds,
            pymapf_timeout=args.pymapf_timeout,
            cuplan_timeout=args.cuplan_timeout,
            pymapf_max_size=args.pymapf_max_size,
            progress=say,
        )
    print("Sweep complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
