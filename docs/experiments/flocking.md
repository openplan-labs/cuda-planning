# Flocking

Boids (Reynolds 1987) with the gains and limits of pymapf's `Boids`,
the library's most GPU-friendly family: one thread per agent, a
brute-force O(n²) neighbour scan, and almost nothing for the host to
do. Conditions for the page: agents start uniformly in a square
scaled to hold density constant, 200 timesteps, seeds {0, 1, 2},
medians with min–max bands. The CPU axis stops at 2048 agents (the
O(n²) pairwise NumPy temporaries are the limit); CUDA continues to
8192. pymapf's swarm module shares the sequential-update semantics
discussed on the [velocity obstacles page](velocity-obstacles.md), so
it is likewise not wall-clock-compared.

## Scaling

![Flocking wall time vs agents, CPU and CUDA](../assets/experiments/flocking-scaling.png#only-light)
![Flocking wall time vs agents, CPU and CUDA](../assets/experiments/flocking-scaling-dark.png#only-dark)

*Median wall time for 200 steps, log–log, 64–8192 agents.*

| agents | cuplan CPU | cuplan CUDA | ratio |
| --: | --: | --: | --: |
| 64 | 0.090 s | **0.045 s** | 2.0× |
| 256 | 1.30 s | **0.105 s** | 12× |
| 1024 | 24.9 s | **0.256 s** | 97× |
| 2048 | 92.2 s | **0.377 s** | **245×** |
| 4096 | — (capped) | 0.819 s | — |
| 8192 | — (capped) | 2.65 s | — |

This is the largest speedup in the library, and the shape explains
why. The CPU materializes every pairwise offset as an `(n, n, 2)`
array — at 2048 agents that is six ~100 MB temporaries per step,
so the line climbs as O(n²) with a cache-miss constant on top. The
CUDA kernel gives each agent one thread that scans the swarm from
registers: work is the same O(n²), but there are no temporaries at
all, and until a few thousand agents the A2000 simply is not full —
visible in the [throughput panel](phases.md) as *rising*
agent-steps/s, peaking around 1.09 M at 2048 agents. Past
saturation the per-thread scan length keeps growing, so throughput
eases back down (0.62 M at 8192) and the curve settles into
parallel O(n²) growth.

Even the smallest swarm measured (64 agents) is already past this
family's CPU/CUDA crossover — a 200-step run amortizes launch
overhead 200 times, unlike a single batched-BFS build.

The raw CSV records the Vicsek polarization at the final step as a
behaviour check, and it is worth being precise about what agreement
means here. Both backends implement the same synchronous update, but
they sum the neighbour contributions in different orders, and 200
steps of a chaotic system amplify that. On 11 of the 12 instances
both backends ran, the final polarizations agree to a relative
difference under 1.3 × 10⁻⁵; on the twelfth (2048 agents, seed 0)
they differ by 1.1%. That is divergence from floating-point
associativity, not a semantic difference — unlike the MAPF families,
where the two backends return
[bit-identical plans](summary.md#headline) because the outputs are
integers. If you need reproducibility across backends in this family,
pin the backend rather than the seed.
