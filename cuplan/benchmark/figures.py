"""Render the Experiments figures from sweep CSVs, in light and dark.

Reads the checkpointed CSVs written by :mod:`cuplan.benchmark.sweep`
and produces every chart on the docs Experiments pages, following the
org's figure rules (``branding/brand/figures.md``): the series being
argued for takes the path accent, supporting series take the agent
ramp, timeouts are drawn as their own mark rather than extrapolated,
and each line shows the median over seeds with a min-max band.

Each figure is written twice — ``<name>.png`` styled by
``frontier.mplstyle`` and ``<name>-dark.png`` with the brand's dark
tokens — so MkDocs Material can swap them with ``#only-light`` /
``#only-dark``.

Run ``python -m cuplan.benchmark.figures --help``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .sweep import SweepRecord, read_records

__all__ = ["render_all"]

# Brand tokens (branding/tokens/tokens.json). Light values match
# frontier.mplstyle; dark values are the same tokens' dark scheme.
LIGHT = {
    "bg": "#f2f4f6", "panel": "#ffffff", "line": "#d9dfe4",
    "heading": "#14181c", "body": "#3c464e", "muted": "#5e6a73",
    "faint": "#7b8790", "path": "#c2472c", "expanded": "#6d8298",
    "frontier": "#127a78",
}
DARK = {
    "bg": "#12171a", "panel": "#1a2126", "line": "#2a343a",
    "heading": "#e8edf0", "body": "#aab6bd", "muted": "#7d8b93",
    "faint": "#59666e", "path": "#e87a5c", "expanded": "#8ba2b7",
    "frontier": "#3fb3ad",
}
AGENT_RAMP = [
    "#3d6d8f", "#4f8a7b", "#7a6f9c", "#8a6d5a",
    "#5b7f9e", "#6b9a8b", "#8f7fae", "#9c7f6a",
]

_MPLSTYLE_URL = (
    "https://raw.githubusercontent.com/openplan-labs/branding/main/"
    "tokens/frontier.mplstyle"
)


def _series_colors(tokens: dict) -> dict:
    # cuplan-cuda is the argued-for series -> path accent.
    return {
        "cuplan-cuda": tokens["path"],
        "cuplan-cpu": AGENT_RAMP[0],
        "pymapf": AGENT_RAMP[2],
    }


_SERIES_LABEL = {
    "cuplan-cuda": "cuplan CUDA",
    "cuplan-cpu": "cuplan CPU",
    "pymapf": "pymapf",
}
_SERIES_ORDER = ["pymapf", "cuplan-cpu", "cuplan-cuda"]


def _use_style(dark: bool) -> dict:
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    here = Path(__file__).resolve()
    for candidate in [
        Path.cwd() / "tokens" / "frontier.mplstyle",
        here.parents[3] / "branding" / "tokens" / "frontier.mplstyle",
    ]:
        if candidate.exists():
            plt.style.use(str(candidate))
            break
    else:  # pragma: no cover - network fallback
        try:
            plt.style.use(_MPLSTYLE_URL)
        except Exception:
            pass
    tokens = DARK if dark else LIGHT
    if dark:
        plt.rcParams.update(
            {
                "figure.facecolor": tokens["bg"],
                "figure.edgecolor": tokens["bg"],
                "savefig.facecolor": tokens["bg"],
                "savefig.edgecolor": tokens["bg"],
                "axes.facecolor": tokens["panel"],
                "axes.edgecolor": tokens["line"],
                "axes.labelcolor": tokens["body"],
                "axes.titlecolor": tokens["heading"],
                "grid.color": tokens["line"],
                "xtick.color": tokens["muted"],
                "ytick.color": tokens["muted"],
                "text.color": tokens["body"],
            }
        )
    return tokens


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _by(records, **match):
    out = []
    for r in records:
        if all(getattr(r, k) == v for k, v in match.items()):
            out.append(r)
    return out


def _line_stats(rows: list[SweepRecord]):
    """(xs, medians, mins, maxs, timeout_xs, cap) over solved seeds."""
    groups: dict[int, list[SweepRecord]] = defaultdict(list)
    for r in rows:
        groups[r.n_agents].append(r)
    xs, med, lo, hi, t_xs, cap = [], [], [], [], [], None
    for n in sorted(groups):
        cell = groups[n]
        solved = [r.runtime for r in cell if r.status == "solved"]
        timeouts = [r for r in cell if r.status == "timeout"]
        if solved:
            xs.append(n)
            med.append(statistics.median(solved))
            lo.append(min(solved))
            hi.append(max(solved))
        elif timeouts:
            t_xs.append(n)
            cap = timeouts[0].runtime
    return xs, med, lo, hi, t_xs, cap


def _plot_series(ax, rows, solver, colors, band=True):
    xs, med, lo, hi, t_xs, cap = _line_stats(
        [r for r in rows if r.solver == solver]
    )
    color = colors[solver]
    if xs:
        ax.plot(
            xs, med, marker="o", label=_SERIES_LABEL[solver], color=color
        )
        if band and len(xs) > 1:
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
    if t_xs and cap:
        ax.plot(
            t_xs,
            [cap] * len(t_xs),
            marker="^",
            mfc="none",
            ls="none",
            color=color,
            label=f"{_SERIES_LABEL[solver]} (timeout)",
        )


def _log_axes(ax, xlabel, ylabel):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


_LOST = "CUDA arm lost — device fault"

# Statuses that mean "this cell was actually attempted and reported".
_MEASURED = ("solved", "unsolved", "timeout")


def _note_lost(ax, rows, tokens, solver="cuplan-cuda", text=_LOST):
    """Mark a panel whose CUDA rows are all device errors.

    The sweep's MAPF stage was interrupted by a driver fault partway
    through (see the methodology page); the affected cells are on disk
    as ``error`` rows. Drawing nothing there would read as "not
    measured yet", so say which arm is missing instead.
    """
    have = [r for r in rows if r.solver == solver]
    if have and all(r.status == "error" for r in have):
        ax.text(
            0.5, 0.04, text, transform=ax.transAxes, ha="center",
            fontsize=8, color=tokens["muted"], style="italic",
        )


def _measured_sizes(records, family, density):
    """Grid sizes with at least one reported cell at this density."""
    return sorted(
        {
            r.size
            for r in records
            if r.family == family
            and r.density == density
            and r.status in _MEASURED
        }
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


_FAMILY_TITLE = {"prioritized": "Prioritized planning", "pibt": "PIBT"}


def _fig_family_scaling(records, family, colors, tokens, plt, density=0.05):
    """Wall time vs agents, one panel per measured grid, fixed density.

    5% is the sweep's default panel density because it is the only one
    reached on every grid before the run was interrupted; the density
    axis itself is the companion figure.
    """
    sizes = _measured_sizes(records, family, density)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(3.9 * len(sizes), 3.9), sharey=True
    )
    for ax, size in zip(np.atleast_1d(axes), sizes, strict=True):
        rows = _by(records, family=family, size=size, density=density)
        for solver in _SERIES_ORDER:
            _plot_series(ax, rows, solver, colors)
        _log_axes(
            ax, "agents",
            "wall time (s, log)" if size == sizes[0] else "",
        )
        ax.set_title(f"{size}×{size} grid")
        _note_lost(ax, rows, tokens)
        if size == sizes[0]:
            ax.legend()
    fig.suptitle(
        f"{_FAMILY_TITLE[family]} — median wall time, "
        f"{density:.0%} obstacle density",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _fig_family_density(records, family, colors, tokens, plt):
    densities = [0.05, 0.15, 0.25]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    for ax, density in zip(axes, densities, strict=True):
        rows = _by(records, family=family, size=64, density=density)
        for solver in _SERIES_ORDER:
            _plot_series(ax, rows, solver, colors)
        _log_axes(ax, "agents", "wall time (s, log)" if density == 0.05 else "")
        ax.set_title(f"{density:.0%} obstacles")
        _note_lost(ax, rows, tokens)
        if density == densities[0]:
            ax.legend()
    fig.suptitle(
        f"{_FAMILY_TITLE[family]} — density effect on a 64×64 grid",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _fig_success(records, family, colors, tokens, plt, sizes=(32, 64)):
    """Success rate over agents x density, one row per grid.

    A cell is a rate over the seeds that *reported* (solved, unsolved
    or timeout). Cells with nothing to report are drawn as an em dash,
    which covers both deliberate skips and the CUDA rows lost to the
    device fault — the two are separated in the caption, not here.
    """
    from matplotlib.colors import LinearSegmentedColormap

    densities = [0.05, 0.15, 0.25]
    solvers = list(_SERIES_ORDER)
    cmap = LinearSegmentedColormap.from_list(
        "success", [tokens["path"], tokens["expanded"]]
    )
    fig, axes = plt.subplots(
        len(sizes), 3, figsize=(11.4, 3.2 * len(sizes)), squeeze=False
    )
    for row, size in enumerate(sizes):
        agent_axis = sorted(
            {
                r.n_agents
                for r in _by(records, family=family, size=size)
                if r.status in _MEASURED
            }
        )
        for col, solver in enumerate(solvers):
            ax = axes[row][col]
            matrix = np.full((len(densities), len(agent_axis)), np.nan)
            for i, density in enumerate(densities):
                for j, n in enumerate(agent_axis):
                    cell = [
                        r
                        for r in _by(
                            records,
                            family=family,
                            size=size,
                            density=density,
                            solver=solver,
                            n_agents=n,
                        )
                        if r.status in _MEASURED
                    ]
                    if cell:
                        matrix[i, j] = sum(
                            r.status == "solved" for r in cell
                        ) / len(cell)
            ax.imshow(
                np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0,
                aspect="auto", interpolation="nearest",
            )
            for i in range(len(densities)):
                for j in range(len(agent_axis)):
                    value = matrix[i, j]
                    if np.isnan(value):
                        ax.text(
                            j, i, "—", ha="center", va="center",
                            fontsize=8, color=tokens["faint"],
                        )
                    else:
                        ax.text(
                            j, i, f"{value:.0%}",
                            ha="center", va="center", fontsize=8,
                            color=tokens["panel"] if value < 0.7
                            else tokens["heading"],
                        )
            ax.set_xticks(range(len(agent_axis)), [str(a) for a in agent_axis])
            ax.set_yticks(
                range(len(densities)), [f"{d:.0%}" for d in densities]
            )
            ax.set_xlabel("agents")
            if col == 0:
                ax.set_ylabel(f"{size}×{size}\nobstacle density")
            if row == 0:
                ax.set_title(_SERIES_LABEL[solver])
            ax.grid(False)
    fig.suptitle(
        f"{_FAMILY_TITLE[family]} — success rate over 3 seeds "
        "(— = no seed reported)",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


_QUALITY_MARKERS = {"prioritized": "o", "pibt": "s"}
_QUALITY_COLORS = {"prioritized": AGENT_RAMP[0], "pibt": AGENT_RAMP[1]}


def _paired(records, family, metric, left, right):
    """(left, right) metric pairs on instances both solvers solved."""
    solved: dict[tuple, dict] = defaultdict(dict)
    for r in _by(records, family=family):
        if r.status == "solved" and getattr(r, metric) is not None:
            key = (r.size, r.n_agents, r.density, r.seed)
            solved[key][r.solver] = getattr(r, metric)
    pairs = [
        (per[left], per[right])
        for per in solved.values()
        if left in per and right in per
    ]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _fig_quality(records, colors, tokens, plt, reference="cuplan-cpu"):
    """Parity of cuplan against pymapf, plus the backend identity check.

    The scatters use ``cuplan-cpu`` because it covers every instance
    pymapf also solved; the third panel is the licence to read them as
    "cuplan", by showing CPU and CUDA agree exactly wherever both ran.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, metric, label in zip(
        axes[:2],
        ["cost", "makespan"],
        ["sum of costs", "makespan"],
        strict=True,
    ):
        top = 0
        for family in ("prioritized", "pibt"):
            xs, ys = _paired(records, family, metric, "pymapf", reference)
            if not xs:
                continue
            top = max(top, max(xs), max(ys))
            ax.scatter(
                xs, ys, s=18, marker=_QUALITY_MARKERS[family],
                facecolors="none", edgecolors=_QUALITY_COLORS[family],
                label=f"{family} ({len(xs)} instances)", linewidths=1.2,
            )
        lim = [0, top * 1.05 if top else 1]
        ax.plot(lim, lim, color=tokens["faint"], lw=1.0, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"pymapf {label}")
        ax.set_ylabel(f"cuplan {label}")
        ax.set_title(f"{label} vs pymapf")
        ax.legend()

    ax = axes[2]
    top = 0
    for family in ("prioritized", "pibt"):
        xs, ys = _paired(
            records, family, "cost", "cuplan-cpu", "cuplan-cuda"
        )
        if not xs:
            continue
        top = max(top, max(xs), max(ys))
        identical = sum(a == b for a, b in zip(xs, ys, strict=True))
        ax.scatter(
            xs, ys, s=18, marker=_QUALITY_MARKERS[family],
            facecolors="none", edgecolors=_QUALITY_COLORS[family],
            label=f"{family} ({identical}/{len(xs)} identical)",
            linewidths=1.2,
        )
    lim = [0, top * 1.05 if top else 1]
    ax.plot(lim, lim, color=tokens["path"], lw=1.0, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("cuplan CPU sum of costs")
    ax.set_ylabel("cuplan CUDA sum of costs")
    ax.set_title("backend identity check")
    ax.legend()

    fig.suptitle(
        "Solution quality on identical seeded instances "
        "(diagonal = equal cost)",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _fig_bfs_scaling(records, colors, plt):
    sizes = sorted({r.size for r in _by(records, family="bfs")})
    fig, axes = plt.subplots(1, len(sizes), figsize=(11.4, 3.8), sharey=True)
    for ax, size in zip(np.atleast_1d(axes), sizes, strict=True):
        rows = _by(records, family="bfs", size=size)
        for solver in ("cuplan-cpu", "cuplan-cuda"):
            _plot_series(ax, rows, solver, colors)
        _log_axes(
            ax, "distance maps per batch",
            "wall time (s, log)" if size == sizes[0] else "",
        )
        ax.set_title(f"{size}×{size} grid")
        if size == sizes[0]:
            ax.legend()
    fig.suptitle(
        "Batched BFS distance maps — median build time",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _throughput_stats(rows, key="agent_steps_per_s"):
    groups = defaultdict(list)
    for r in rows:
        if r.status == "solved" and key in r.extra:
            groups[r.n_agents].append(r.extra[key])
    xs = sorted(groups)
    return (
        xs,
        [statistics.median(groups[x]) for x in xs],
        [min(groups[x]) for x in xs],
        [max(groups[x]) for x in xs],
    )


def _fig_throughput(bfs, vo, flocking, colors, plt):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    panels = [
        (
            axes[0],
            _by(bfs, family="bfs", size=256),
            "maps_per_s",
            "distance maps per batch",
            "maps / s",
            "Batched BFS, 256×256",
        ),
        (
            axes[1],
            _by(vo, family="velocity_obstacles"),
            "agent_steps_per_s",
            "agents",
            "agent-steps / s",
            "Velocity obstacles, 80 steps",
        ),
        (
            axes[2],
            _by(flocking, family="flocking"),
            "agent_steps_per_s",
            "agents",
            "agent-steps / s",
            "Flocking, 200 steps",
        ),
    ]
    for ax, rows, key, xlabel, ylabel, title in panels:
        for solver in ("cuplan-cpu", "cuplan-cuda"):
            xs, med, lo, hi = _throughput_stats(
                [r for r in rows if r.solver == solver], key
            )
            if not xs:
                continue
            color = colors[solver]
            ax.plot(
                xs, med, marker="o", color=color, label=_SERIES_LABEL[solver]
            )
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        _log_axes(ax, xlabel, ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle(
        "Throughput vs batch size — a rising CUDA line means the GPU "
        "is not yet saturated",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _fig_sim_scaling(records, family, title, colors, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rows = _by(records, family=family)
    for solver in ("cuplan-cpu", "cuplan-cuda"):
        _plot_series(ax, rows, solver, colors)
    _log_axes(ax, "agents", "wall time (s, log)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


_PHASE_COLORS = {
    "h2d": AGENT_RAMP[0],
    "kernel": None,  # path accent, set per mode
    "d2h": AGENT_RAMP[2],
    "host": None,  # faint, set per mode
    "oracle": AGENT_RAMP[1],
    "search": None,  # path accent
    "reserve": AGENT_RAMP[2],
}

# Colour alone would separate these four blues badly in greyscale and
# for a colour-blind reader; hatching is the second channel.
_PHASE_HATCH = {
    "h2d": "///",
    "kernel": "",
    "d2h": "...",
    "host": "\\\\\\",
    "oracle": "///",
    "search": "",
    "reserve": "...",
}


def _read_phases(path: Path):
    import csv

    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": row["family"],
                    "size": int(row["size"]),
                    "n": int(row["n"]),
                    "seed": int(row["seed"]),
                    "phase": row["phase"],
                    "seconds": float(row["seconds"]),
                }
            )
    return rows


def _phase_medians(rows, family):
    """{n: {phase: median seconds}} for one family."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["family"] == family:
            per[(r["size"], r["n"])][r["phase"]].append(r["seconds"])
    return {
        key: {p: statistics.median(v) for p, v in phases.items()}
        for key, phases in sorted(per.items())
    }


def _stacked_panel(ax, medians, phases, tokens, xlabel, labeler=None):
    colors = dict(_PHASE_COLORS)
    colors["kernel"] = tokens["path"]
    colors["search"] = tokens["path"]
    colors["host"] = tokens["faint"]
    keys = list(medians)
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for phase in phases:
        values = np.array(
            [
                medians[k].get(phase, 0.0)
                / max(sum(medians[k].values()), 1e-12)
                for k in keys
            ]
        )
        ax.bar(
            x, values, bottom=bottom, width=0.62, label=phase,
            color=colors[phase], hatch=_PHASE_HATCH[phase],
            edgecolor=tokens["panel"], linewidth=0.0,
        )
        bottom += values
    for i, k in enumerate(keys):
        total = sum(medians[k].values())
        ax.text(
            i, 1.03, f"{total:.2f}s", ha="center", fontsize=8,
            color=tokens["muted"],
        )
    labeler = labeler or (lambda k: str(k[1]))
    ax.set_xticks(x, [labeler(k) for k in keys])
    ax.set_ylim(0, 1.14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of wall time")


def _fig_phases(phase_rows, tokens, plt):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharey=True)
    panels = [
        ("bfs", "distance maps per batch", "Batched BFS, 256×256"),
        ("velocity_obstacles", "agents", "Velocity obstacles, 80 steps"),
        ("flocking", "agents", "Flocking, 200 steps"),
    ]
    for ax, (family, xlabel, title) in zip(axes, panels, strict=True):
        medians = _phase_medians(phase_rows, family)
        if not medians:
            continue
        _stacked_panel(
            ax, medians, ["kernel", "h2d", "d2h", "host"], tokens, xlabel
        )
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Where CUDA wall time goes — medians over 3 seeds "
        "(profiled runs; totals above bars)",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def _fig_prioritized_phases(phase_rows, tokens, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    medians = _phase_medians(phase_rows, "prioritized")
    if medians:
        _stacked_panel(
            ax,
            medians,
            ["search", "oracle", "reserve", "host"],
            tokens,
            "grid, agents",
            labeler=lambda k: f"{k[0]}², {k[1]}",
        )
    ax.set_title("Prioritized planning on CUDA — solver-level split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fig.tight_layout()
    return fig


def _speedup(rows, fast, slow):
    groups = defaultdict(dict)
    for r in rows:
        if r.status == "solved":
            groups[(r.n_agents, r.seed)][r.solver] = r.runtime
    per_n = defaultdict(list)
    for (n, _seed), per in groups.items():
        if fast in per and slow in per:
            per_n[n].append(per[slow] / per[fast])
    xs = sorted(per_n)
    return xs, [statistics.median(per_n[x]) for x in xs]


def _fig_crossover(mapf, bfs, vo, flocking, tokens, plt):
    """Speedup ratios per family — the one summary figure.

    MAPF grids differ between the two panels on purpose: the CUDA arm
    only survives at 5% density (the fault took the rest), while the
    CPU-vs-pymapf comparison has the full density axis at 15%. Each
    panel says which grid and density it is drawn from.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    device = [
        (
            _by(mapf, family="prioritized", size=64, density=0.05),
            "prioritized 64², 5%",
            AGENT_RAMP[0],
        ),
        (
            _by(mapf, family="pibt", size=64, density=0.05),
            "PIBT 64², 5%",
            AGENT_RAMP[1],
        ),
        (_by(bfs, family="bfs", size=256), "BFS 256²", AGENT_RAMP[2]),
        (
            _by(vo, family="velocity_obstacles"),
            "velocity obstacles",
            AGENT_RAMP[3],
        ),
        (_by(flocking, family="flocking"), "flocking", AGENT_RAMP[5]),
    ]
    ax = axes[0]
    for rows, label, color in device:
        xs, ratio = _speedup(rows, "cuplan-cuda", "cuplan-cpu")
        if xs:
            ax.plot(xs, ratio, marker="o", label=label, color=color)
    ax.axhline(1.0, color=tokens["faint"], lw=1.0)
    _log_axes(ax, "agents / batch size", "CPU time ÷ CUDA time (log)")
    ax.set_title("CUDA speedup over cuplan CPU")
    ax.legend()

    ax = axes[1]
    for rows, label, color in [
        (
            _by(mapf, family="prioritized", size=64, density=0.15),
            "prioritized 64², 15%",
            AGENT_RAMP[0],
        ),
        (
            _by(mapf, family="pibt", size=64, density=0.15),
            "PIBT 64², 15%",
            AGENT_RAMP[1],
        ),
    ]:
        xs, ratio = _speedup(rows, "cuplan-cpu", "pymapf")
        if xs:
            ax.plot(xs, ratio, marker="o", label=label, color=color)
    ax.axhline(1.0, color=tokens["faint"], lw=1.0)
    _log_axes(ax, "agents", "pymapf time ÷ cuplan CPU time (log)")
    ax.set_title("cuplan CPU speedup over pymapf")
    ax.legend()
    fig.suptitle(
        "Crossovers — above the line the denominator wins "
        "(medians over seeds solved by both; timeouts excluded)",
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def _fig_headline(mapf, colors, tokens, plt):
    """One panel for the README: prioritized planning, three solvers."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = _by(mapf, family="prioritized", size=64, density=0.05)
    for solver in _SERIES_ORDER:
        _plot_series(ax, rows, solver, colors)
    _log_axes(ax, "agents", "wall time (s, log)")
    ax.set_title(
        "Prioritized planning — 64×64 grid, 5% obstacles, "
        "median of 3 seeds"
    )
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def render_all(
    data: Path, out_dirs: list[Path], progress=None
) -> list[Path]:
    """Render every figure into each of ``out_dirs``; returns paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    say = progress or (lambda s: None)
    mapf = read_records(data / "mapf.csv") if (data / "mapf.csv").exists() else []
    bfs = read_records(data / "bfs.csv") if (data / "bfs.csv").exists() else []
    vo = read_records(data / "vo.csv") if (data / "vo.csv").exists() else []
    flocking = (
        read_records(data / "flocking.csv")
        if (data / "flocking.csv").exists()
        else []
    )
    phase_rows = (
        _read_phases(data / "phases.csv")
        if (data / "phases.csv").exists()
        else []
    )

    written: list[Path] = []
    for dark in (False, True):
        tokens = _use_style(dark)
        colors = _series_colors(tokens)
        figures: dict = {}
        if mapf:
            for family in ("prioritized", "pibt"):
                figures[f"{family}-scaling"] = _fig_family_scaling(
                    mapf, family, colors, tokens, plt
                )
                figures[f"{family}-density"] = _fig_family_density(
                    mapf, family, colors, tokens, plt
                )
                figures[f"{family}-success"] = _fig_success(
                    mapf, family, colors, tokens, plt
                )
            figures["quality-parity"] = _fig_quality(
                mapf, colors, tokens, plt
            )
            figures["headline"] = _fig_headline(mapf, colors, tokens, plt)
        if bfs:
            figures["bfs-scaling"] = _fig_bfs_scaling(bfs, colors, plt)
        if vo:
            figures["vo-scaling"] = _fig_sim_scaling(
                vo,
                "velocity_obstacles",
                "Velocity obstacles — 80 steps, antipodal circle",
                colors,
                plt,
            )
        if flocking:
            figures["flocking-scaling"] = _fig_sim_scaling(
                flocking,
                "flocking",
                "Flocking — 200 steps, uniform square start",
                colors,
                plt,
            )
        if bfs and vo and flocking:
            figures["throughput"] = _fig_throughput(
                bfs, vo, flocking, colors, plt
            )
        if phase_rows:
            figures["phases"] = _fig_phases(phase_rows, tokens, plt)
            figures["prioritized-phases"] = _fig_prioritized_phases(
                phase_rows, tokens, plt
            )
        if mapf and bfs and vo and flocking:
            figures["crossover"] = _fig_crossover(
                mapf, bfs, vo, flocking, tokens, plt
            )
        suffix = "-dark" if dark else ""
        for name, fig in figures.items():
            for out_dir in out_dirs:
                if dark and out_dir.name == "figures":
                    continue  # benchmarks/ archive keeps light only
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{name}{suffix}.png"
                fig.savefig(path)
                written.append(path)
                say(str(path))
            plt.close(fig)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuplan.benchmark.figures",
        description="Render Experiments figures from sweep CSVs.",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("benchmarks/experiments")
    )
    parser.add_argument(
        "--out",
        type=Path,
        nargs="+",
        default=[
            Path("docs/assets/experiments"),
            Path("benchmarks/experiments/figures"),
        ],
    )
    args = parser.parse_args(argv)
    render_all(args.data, args.out, progress=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
