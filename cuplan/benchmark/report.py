"""Turn benchmark records into CSV, Markdown, and Frontier-styled charts."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from .harness import BenchmarkResult, machine_description

__all__ = ["write_csv", "write_markdown", "write_charts"]

# Frontier brand: the solver being argued for takes the path accent,
# supporting series take the agent ramp (brand/figures.md).
_SERIES_COLORS = {
    "cuplan-cuda": "#c2472c",  # --color-path: the argued-for series
    "cuplan-cpu": "#3d6d8f",  # agent ramp 1
    "pymapf": "#7a6f9c",  # agent ramp 3
}
_MPLSTYLE_URL = (
    "https://raw.githubusercontent.com/openplan-labs/branding/main/"
    "tokens/frontier.mplstyle"
)


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    """Write the flat records as CSV (one row per measurement)."""
    fields = [
        "family",
        "solver",
        "size",
        "n_agents",
        "seed",
        "success",
        "runtime",
        "cost",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in results:
            writer.writerow(record.as_dict())


def _aggregate(
    results: list[BenchmarkResult],
) -> dict[tuple[str, str, int, int], dict]:
    """Group by (family, solver, size, n_agents); median over seeds."""
    groups: dict[tuple[str, str, int, int], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        groups[(r.family, r.solver, r.size, r.n_agents)].append(r)
    out = {}
    for key, rows in groups.items():
        solved = [r for r in rows if r.success]
        out[key] = {
            "runs": len(rows),
            "success_rate": len(solved) / len(rows),
            "median_runtime": (
                statistics.median(r.runtime for r in solved) if solved else None
            ),
            "median_cost": (
                statistics.median(r.cost for r in solved if r.cost is not None)
                if solved
                else None
            ),
        }
    return out


def write_markdown(results: list[BenchmarkResult], path: Path) -> None:
    """Write an aggregated Markdown table with the measurement conditions."""
    agg = _aggregate(results)
    lines = [
        "# Benchmark results",
        "",
        f"Machine: {machine_description()}. Median over seeds; wall time",
        "covers the full solve including host/device transfers. Random",
        "grids at 15% obstacle density; identical instances for every",
        "solver. Reproduce with `python -m cuplan.benchmark`.",
        "",
        "| family | solver | grid | agents | success | median time (s) | median cost |",
        "| :-- | :-- | --: | --: | --: | --: | --: |",
    ]
    for key in sorted(agg):
        family, solver, size, n_agents = key
        row = agg[key]
        time_s = (
            f"{row['median_runtime']:.4f}"
            if row["median_runtime"] is not None
            else "—"
        )
        cost = (
            f"{row['median_cost']:.0f}" if row["median_cost"] is not None else "—"
        )
        grid = f"{size}x{size}" if size else "—"
        lines.append(
            f"| {family} | {solver} | {grid} | {n_agents} | "
            f"{row['success_rate']:.0%} | {time_s} | {cost} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _style():
    import matplotlib.pyplot as plt

    style = Path(__file__).resolve()
    # Prefer a local branding checkout when present (CI-friendly),
    # else the published stylesheet.
    for candidate in [
        Path.cwd() / "tokens" / "frontier.mplstyle",
        style.parents[3] / "branding" / "tokens" / "frontier.mplstyle",
    ]:
        if candidate.exists():
            plt.style.use(str(candidate))
            return
    try:
        plt.style.use(_MPLSTYLE_URL)
    except Exception:
        pass  # unstyled is better than no chart


def write_charts(results: list[BenchmarkResult], out_dir: Path) -> list[Path]:
    """Render runtime-scaling charts per family. Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    agg = _aggregate(results)
    written: list[Path] = []

    families = sorted({k[0] for k in agg})
    for family in families:
        sizes = sorted({k[2] for k in agg if k[0] == family})
        largest = sizes[-1] if sizes else 0
        keys = [k for k in agg if k[0] == family and k[2] == largest]
        solvers = sorted({k[1] for k in keys})
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for solver in solvers:
            points = sorted(
                (k[3], agg[k]["median_runtime"])
                for k in keys
                if k[1] == solver and agg[k]["median_runtime"] is not None
            )
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(
                xs,
                ys,
                marker="o",
                label=solver,
                color=_SERIES_COLORS.get(solver),
            )
        ax.set_yscale("log")
        ax.set_xlabel(
            "sources (one BFS per agent)" if family == "bfs" else "agents"
        )
        ax.set_ylabel("median wall time (s, log scale)")
        grid_note = f" — {largest}x{largest} grid" if largest else ""
        ax.set_title(f"{family}{grid_note}")
        ax.legend()
        path = out_dir / f"{family}-scaling.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written
