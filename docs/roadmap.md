# Roadmap

Planned solvers exist today as documented stubs in `cuplan.roadmap` —
they raise `NotImplementedError` with a pointer here, so nothing can
mistake a stub for a solver. pymapf has working CPU implementations of
all of them, which fixes the semantics before any kernel is written.

| Algorithm | Paper | Intended GPU strategy |
| :-- | :-- | :-- |
| **CBS** | Sharon, Stern, Felner and Sturtevant 2015, AIJ 219 | The high-level constraint tree is sequential, but sibling nodes' low-level searches are independent: batch them as parallel space-time wavefronts. |
| **LaCAM** | Okumura 2023, AAAI | Reuse cuplan's PIBT step (batched candidate evaluation) as the configuration generator; the lazy high-level DFS stays on the host. Restores the completeness PIBT lacks. |
| **MAPF-LNS** | Li, Chen, Harabor, Stuckey and Koenig 2021, IJCAI | Destroy/repair proposals are independent — evaluate many neighbourhoods concurrently, keep the best repair. |
| **SIPP** | Phillips and Likhachev 2011, ICRA | Safe-interval construction from a reservation table is a per-cell scan (one thread per cell); the interval-graph search stays on the host. |
| **NMPC** | mirrors pymapf's decentralized NMPC agent | Sampling-based MPC (MPPI): thousands of rollouts per agent per step, one thread each. |

Also on the list, smaller:

- Diagonal moves (8-connected grids) across the wavefront searches.
- Keeping solver outputs on-device end to end for pipelines that
  consume them there.
- Multi-GPU batching for benchmark-scale scenario sweeps.

If you want one of these, open an issue first — the approach is worth
agreeing on before anyone invests in kernels. Contributions welcome;
see
[CONTRIBUTING.md](https://github.com/openplan-labs/cuda-planning/blob/main/CONTRIBUTING.md).
