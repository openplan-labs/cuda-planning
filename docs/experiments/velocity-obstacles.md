# Velocity obstacles

Decentralized collision avoidance (Fiorini and Shiller 1998), the
library's smallest-kernel family and therefore the one with the most
visible crossover. Conditions for the page: agents on a circle with
antipodal goals (the all-cross stress case), 80 timesteps, 20 × 5
velocity samples per agent, seeds {0, 1, 2}, medians with min–max
bands. pymapf is absent by design: its simulator updates agents
sequentially *within* a timestep, so a wall-clock comparison would
time a different problem — see the
[semantics note](../algorithms/velocity-obstacles.md).

## Scaling

![Velocity obstacles wall time vs agents, CPU and CUDA](../assets/experiments/vo-scaling.png#only-light)
![Velocity obstacles wall time vs agents, CPU and CUDA](../assets/experiments/vo-scaling-dark.png#only-dark)

*Median wall time for the 80-step run, log–log, 8–1024 agents. The
CPU axis stops at 1024 agents (its O(n² × samples) temporaries are the
limit, not patience).*

| agents | cuplan CPU | cuplan CUDA | ratio |
| --: | --: | --: | --: |
| 8 | **0.016 s** | 0.029 s | 0.6× |
| 16 | **0.035 s** | 0.037 s | 0.9× |
| 32 | 0.114 s | **0.059 s** | 1.9× |
| 128 | 1.69 s | **0.431 s** | 3.9× |
| 512 | 31.2 s | **5.57 s** | 5.6× |
| 1024 | 115.8 s | **21.9 s** | 5.3× |

The curve bends exactly where the arithmetic says it should. One step
evaluates `agents² × samples` collision cones — at 8 agents, with a
20 × 5 velocity sample grid, that is 6 400 tests, far too little to
pay for 80 × 3 kernel launches plus transfers, and the CPU wins.
**The crossover sits between 16 and 32 agents** (16 is a dead heat:
34.7 ms vs 37.1 ms). From there both backends grow O(n²) — the
parallelism is over agents and samples, but each thread's scan over
the others grows linearly — so the ratio plateaus around 5–6× rather
than growing without bound. The [phase breakdown](phases.md) shows
the same story from the inside: at 16 agents the kernel is only 60%
of CUDA wall time; at 1024 agents it is 99.8%.

Two caveats the numbers force. First, 80 steps is a *compute budget*,
not a task: the start circle's radius grows with the agent count, so
beyond 8 agents the goals are farther than 80 steps can cover at
`vmax`, and `goals_reached` is 0 for every run from 16 agents up.
That column measures the budget, not solver failure.

Second, and more seriously, **plain velocity obstacles do not keep
the agents apart in this scenario, at any size measured.** With
`radius = 0.5` the contact distance is 1.0; the median
`min_separation` over seeds is 0.74 at 8 agents, 0.087 at 16, 0.045
at 64, 0.0097 at 256 and 0.0005 at 512. Agents are interpenetrating
from 16 agents onward. This is the known failure of the
non-reciprocal formulation — every agent assumes the others hold
their velocity, so two agents facing each other both dodge the same
way and the avoidance cancels. Reciprocal variants (ORCA) are on the
[roadmap](../roadmap.md). Read this page as a throughput measurement
of a kernel, not as a claim that the kernel produces safe motion at
scale.

What the run does establish besides speed is backend equivalence: on
all 24 instances CPU and CUDA report *bit-identical* `goals_reached`
and `min_separation`, because the synchronous update rule makes the
step order-independent.
