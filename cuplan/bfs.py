"""Batched grid BFS: one flood-fill distance map per source.

This is the workhorse primitive of the library. MAPF solvers consume
exact goal-distance tables — pymapf computes one backward Dijkstra per
agent, serially. On a 4-connected grid with unit edge costs Dijkstra
*is* BFS, and BFS is a wavefront: every cell on the frontier can be
expanded simultaneously. Batching N sources into one ``(N, H, W)``
volume turns the whole heuristic-table build into ``makespan`` fully
parallel sweeps.

Reference: Merrill, D.; Garland, M.; and Grimshaw, A. 2012. *Scalable
GPU graph traversal.* PPoPP 2012: 117-128 (frontier-parallel BFS).
"""

from __future__ import annotations

import numpy as np

from .backend import Backend, resolve_backend
from .grid import Grid

__all__ = ["distance_maps"]


def distance_maps(
    grid: Grid,
    sources: np.ndarray,
    backend: Backend = "auto",
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Return exact 4-connected distances from each source to every cell.

    Args:
        grid: the occupancy grid.
        sources: ``(n, 2)`` array of ``(row, col)`` source cells. Each
            must be a free cell.
        backend: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        timings: optional dict the CUDA backend fills with per-phase
            wall times in seconds — ``h2d`` (upload + device
            allocation), ``kernel`` (the wave loop, including the
            per-wave termination check, which is a 4-byte device read),
            and ``d2h`` (copying the finished maps back). The CPU
            backend leaves it untouched. Phase boundaries are device
            synchronization points, so profiling adds a small cost;
            benchmark totals should come from an unprofiled run.

    Returns:
        ``(n, height, width)`` int32 array; entry ``[i, r, c]`` is the
        length of a shortest path from ``sources[i]`` to ``(r, c)``, or
        ``-1`` where unreachable (including blocked cells).
    """
    sources = np.atleast_2d(np.asarray(sources, dtype=np.int32))
    if sources.ndim != 2 or sources.shape[1] != 2:
        raise ValueError("sources must have shape (n, 2)")
    for r, c in sources:
        if not grid.is_free((int(r), int(c))):
            raise ValueError(f"source ({r}, {c}) is blocked or out of bounds")
    if resolve_backend(backend) == "cuda":
        return _distance_maps_cuda(grid, sources, timings)
    return _distance_maps_cpu(grid, sources)


def _distance_maps_cpu(grid: Grid, sources: np.ndarray) -> np.ndarray:
    """Vectorized NumPy wavefront: one array sweep per BFS level."""
    n = len(sources)
    free = grid.free
    dist = np.full((n, grid.height, grid.width), -1, dtype=np.int32)
    frontier = np.zeros_like(dist, dtype=bool)
    frontier[np.arange(n), sources[:, 0], sources[:, 1]] = True
    dist[frontier] = 0

    wave = 0
    while frontier.any():
        wave += 1
        spread = np.zeros_like(frontier)
        spread[:, 1:, :] |= frontier[:, :-1, :]
        spread[:, :-1, :] |= frontier[:, 1:, :]
        spread[:, :, 1:] |= frontier[:, :, :-1]
        spread[:, :, :-1] |= frontier[:, :, 1:]
        frontier = spread & free[None, :, :] & (dist < 0)
        dist[frontier] = wave
    return dist


def _distance_maps_cuda(
    grid: Grid, sources: np.ndarray, timings: dict[str, float] | None = None
) -> np.ndarray:
    """Frontier-parallel BFS on the device: one kernel launch per level."""
    import time

    import cupy

    from .kernels import get_kernel

    profile = timings is not None
    if profile:
        cupy.cuda.get_current_stream().synchronize()
        mark = time.perf_counter()

    kernel = get_kernel("bfs", "bfs_wave")
    n = len(sources)
    cells = grid.height * grid.width
    dist = cupy.full((n, cells), -1, dtype=cupy.int32)
    linear = grid.to_linear(sources)
    dist[cupy.arange(n), cupy.asarray(linear)] = 0
    free_mask = cupy.asarray(grid.free.reshape(-1).astype(np.uint8))
    changed = cupy.zeros(1, dtype=cupy.int32)

    if profile:
        cupy.cuda.get_current_stream().synchronize()
        now = time.perf_counter()
        timings["h2d"] = timings.get("h2d", 0.0) + now - mark
        mark = now

    threads = 256
    blocks = min(65535, (n * cells + threads - 1) // threads)
    wave = 0
    while True:
        wave += 1
        changed[0] = 0
        kernel(
            (blocks,),
            (threads,),
            (
                dist,
                free_mask,
                np.int32(n),
                np.int32(grid.height),
                np.int32(grid.width),
                np.int32(wave),
                changed,
            ),
        )
        if int(changed[0]) == 0:
            break

    if profile:
        now = time.perf_counter()
        timings["kernel"] = timings.get("kernel", 0.0) + now - mark
        mark = now

    result = cupy.asnumpy(dist).reshape(n, grid.height, grid.width)

    if profile:
        timings["d2h"] = (
            timings.get("d2h", 0.0) + time.perf_counter() - mark
        )
    return result
