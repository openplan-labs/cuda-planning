"""Benchmark harness comparing cuplan (CPU and CUDA) with pymapf.

Run it as a module::

    python -m cuplan.benchmark --out benchmarks/results

Every scenario is generated once from a fixed seed and handed to every
solver, so the numbers compare the same problems. Results are written
as CSV and Markdown, with charts styled by the Frontier brand
stylesheet when matplotlib is available.
"""

from .harness import (
    BenchmarkResult,
    run_bfs_benchmark,
    run_mapf_benchmark,
    run_vo_benchmark,
)
from .scenarios import Scenario, random_scenario

__all__ = [
    "BenchmarkResult",
    "Scenario",
    "random_scenario",
    "run_bfs_benchmark",
    "run_mapf_benchmark",
    "run_vo_benchmark",
]
