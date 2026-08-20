"""Command-line entry point: ``python -m cuplan.benchmark``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..backend import cuda_available
from .harness import run_mapf_benchmark, run_vo_benchmark
from .report import write_charts, write_csv, write_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuplan.benchmark",
        description="Benchmark cuplan (CPU and CUDA) against pymapf on "
        "identical scenarios.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results"),
        help="output directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[32, 64, 128],
        help="grid side lengths",
    )
    parser.add_argument(
        "--agents", type=int, nargs="+", default=[8, 32, 64, 128],
        help="agent counts",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="scenario seeds (medians are taken across these)",
    )
    parser.add_argument(
        "--vo-agents", type=int, nargs="+", default=[16, 64, 128],
        help="agent counts for the velocity-obstacle benchmark",
    )
    parser.add_argument(
        "--no-pymapf", action="store_true", help="skip the pymapf baselines"
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="skip chart rendering"
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    say = lambda s: print(f"  {s}", flush=True)  # noqa: E731

    print(f"CUDA available: {cuda_available()}")
    print("MAPF families (prioritized, PIBT):")
    results = run_mapf_benchmark(
        sizes=args.sizes,
        agent_counts=args.agents,
        seeds=args.seeds,
        include_pymapf=not args.no_pymapf,
        progress=say,
    )
    print("Velocity obstacles:")
    results += run_vo_benchmark(
        agent_counts=args.vo_agents, seeds=args.seeds, progress=say
    )

    write_csv(results, args.out / "results.csv")
    write_markdown(results, args.out / "results.md")
    print(f"Wrote {args.out / 'results.csv'} and {args.out / 'results.md'}")
    if not args.no_charts:
        try:
            for path in write_charts(results, args.out):
                print(f"Wrote {path}")
        except ImportError:
            print("matplotlib not installed; skipping charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
