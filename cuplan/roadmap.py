"""Planned solvers: documented stubs, not implementations.

Each class below names the algorithm, the paper, and the intended
parallelization strategy, and raises :class:`NotImplementedError` from
its constructor so nothing can mistake a stub for a solver. pymapf has
working CPU implementations of all of them.
"""

from __future__ import annotations

__all__ = ["CBS", "LaCAM", "LNS", "SIPP", "NMPC"]


class _Planned:
    """Base for planned-but-unimplemented solvers."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is on the cuplan roadmap but not yet "
            "implemented. pymapf has a CPU implementation: "
            "https://github.com/openplan-labs/pymapf"
        )


class CBS(_Planned):
    """Conflict-Based Search (Sharon et al. 2015, AIJ 219:40-66).

    Optimal two-level search. GPU plan: the high-level constraint tree
    is sequential, but sibling nodes' low-level searches are independent
    — batch them as parallel space-time wavefronts, one stream each.
    """


class LaCAM(_Planned):
    """LaCAM (Okumura 2023, AAAI): complete search wrapping PIBT.

    GPU plan: reuse cuplan's PIBT step (batched candidate evaluation);
    the lazy high-level DFS stays on the host.
    """


class LNS(_Planned):
    """MAPF-LNS (Li et al. 2021, IJCAI): large neighbourhood search.

    GPU plan: destroy/repair proposals are independent — evaluate many
    neighbourhoods concurrently and keep the best repair.
    """


class SIPP(_Planned):
    """Safe Interval Path Planning (Phillips and Likhachev 2011, ICRA).

    GPU plan: safe-interval construction from a reservation table is a
    per-cell scan (one thread per cell); the interval graph search
    itself is small enough to stay on the host.
    """


class NMPC(_Planned):
    """Decentralized nonlinear MPC (mirroring pymapf's NMPC agent).

    GPU plan: sampling-based MPC (MPPI) — thousands of rollouts per
    agent per step, each one thread.
    """
