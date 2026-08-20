# PIBT

PIBT — Priority Inheritance with Backtracking (Okumura, Machida,
Défago & Tamura 2022) — is a one-step-at-a-time solver: at every
timestep each agent ranks its five moves by distance-to-goal, and
conflicts are resolved by letting a higher-priority agent *inherit*
its blocker's decision, backtracking when the inheritance chain
fails. There is no path search at all, which makes it very fast and
**incomplete**, and which changes where a GPU can help.

Conditions for the whole page: random grids from `random_scenario`,
starts and goals in one connected component, seeds {0, 1, 2},
identical instances to all three solvers, medians over seeds with
min–max bands, wall time including host/device transfers, caps of
60 s (pymapf) and 300 s (cuplan). Full protocol on the
[methodology page](methodology.md).

## Scaling with agents

![PIBT wall time vs agents on 32x32, 64x64 and 128x128 grids](../assets/experiments/pibt-scaling.png#only-light)
![PIBT wall time vs agents on 32x32, 64x64 and 128x128 grids](../assets/experiments/pibt-scaling-dark.png#only-dark)

*Median wall time, log–log, at **5% obstacle density** — the one
density reached on all three grids before the sweep was interrupted.
Grids 32², 64², 128²; agents 8–512; seeds {0, 1, 2}; bands are
min–max over seeds that solved. The 128² panel carries no CUDA line:
those cells are on disk as device errors, not as measurements.*

PIBT's curves are the flattest in the library, and the flatness is
the finding.

**Against pymapf the ratio is roughly constant.** 7.2× at 8 agents,
9.8× at 32, 10.7× at 64, 11.1× at 256, 9.9× at 512 — all on the 64²
grid at 5%. Compare the [prioritized planning page](prioritized.md),
where the same comparison runs from 1.6× to 78× over a narrower
range. The difference is structural: prioritized planning's cost per
agent grows with how full the reservation table already is, so the
gap between a dense array and a growing set compounds; PIBT does a
fixed amount of work per agent per timestep regardless of history, so
the gap is a constant factor — vectorized NumPy against Python
object churn — and stays one.

**Against its own CPU backend the GPU wins by grid, not by agents.**
On the 64² grid CUDA runs 1.2× at 8 agents rising to 2.2× at 64. On
the 32² grid it is a coin toss: 0.96× at 8 agents, 1.4× at 64, and
0.92× at 128 — the GPU *loses* on the largest 32² cell measured.

The reason is that the backend switch moves exactly one thing. In
PIBT the device builds the [distance oracle](bfs.md) — one batched
BFS, once, before the first timestep — and nothing else: the per-step
candidate gather and the inheritance chains are host NumPy in both
backends, [by construction](../algorithms/pibt.md#parallelization).
So the CUDA margin is capped by the oracle's share of runtime, and
that share is set by grid area. A 32² oracle is 1024 cells per map, a
sub-millisecond job that does not repay handing the grid to a device
and taking `n` maps back; four times the cells at 64², over roughly
twice the BFS diameter, does. Agents make the *host* loop longer
without giving the device more to do, which is why the 32² ratio
falls back below 1× at 128 agents.

So for PIBT, `backend="cpu"` is the right default below roughly a 64²
grid, whatever the agent count — and the ceiling above it is
Amdahl's, not the kernel's.

**Neither backend's curve is superlinear in agents.** PIBT is
O(agents × timesteps) with a small constant, and the timestep count
is bounded by the solver's step limit rather than by search depth.
The 64² CPU line moves from 0.011 s at 8 agents to 0.54 s at 512: a
64× increase in agents for a 51× increase in time.

### Selected medians, 64×64 grid, 5% density

| agents | pymapf | cuplan CPU | cuplan CUDA | CUDA vs CPU | pymapf vs CPU |
| --: | --: | --: | --: | --: | --: |
| 8 | 0.076 s | 0.011 s | **0.0086 s** | 1.2× | 7.2× |
| 16 | 0.154 s | 0.018 s | **0.0123 s** | 1.5× | 8.5× |
| 32 | 0.306 s | 0.031 s | **0.0190 s** | 1.7× | 9.8× |
| 64 | 0.618 s | 0.058 s | **0.0267 s** | 2.2× | 10.7× |
| 128 | 1.296 s | 0.118 s | **0.0625 s** ² | 2.0× | 11.1× |
| 256 | 2.662 s | **0.235 s** | — ¹ | — | 11.1× |
| 512 | 5.348 s | **0.541 s** | — ¹ | — | 9.9× |

*Medians over 3 seeds unless noted. Ratios are medians of per-seed
ratios on instances both solvers solved, so they are not the ratio of
the two median columns.
¹ CUDA rows lost to the device fault — see
[methodology](methodology.md).
² 1 seed of 3; the other two were the first rows the fault touched,
so treat this cell as indicative rather than measured.*

Note what is *not* in this table: a timeout. PIBT finished every
64² cell in every library, up to 512 agents. Its failure mode is
returning "no plan", not running long.

## Density

![PIBT wall time vs agents at 5, 15 and 25 percent obstacle density](../assets/experiments/pibt-density.png#only-light)
![PIBT wall time vs agents at 5, 15 and 25 percent obstacle density](../assets/experiments/pibt-density-dark.png#only-dark)

*64×64 grid, obstacle density 5% / 15% / 25%, agents 8–512, seeds
{0, 1, 2}, shared y-axis. The 15% and 25% panels carry no CUDA line:
the device fault reached the sweep before those cells ran.*

This is the figure where PIBT's shape changes, and it is worth
reading carefully because the y-axis is medians over *solved* seeds
only.

At 5% the CPU line is a clean straight log–log run. At 25% it kinks
upward hard between 32 and 64 agents — 0.034 s to 0.195 s, a 5.7×
jump for a 2× agent increase — and then goes *flat*, 0.255 s at 128
and 0.277 s at 256. That flat tail is a selection effect, not a
plateau: at 128 agents one of three seeds failed and at 256 two of
three failed, and the failing seeds are the slow ones. Per-seed at
(64², 25%, 256 agents) the two failures took 2.21 s and 2.18 s while
the single success took 0.276 s. The median of solved runs hides an
order of magnitude.

The mechanism is priority inheritance. In open space a blocked agent
steps aside and the chain is one link long. Pack agents into a
quarter-blocked grid and the chains lengthen, backtrack, and
eventually exhaust the step limit — so wall time and failure rate
rise together, and quoting either one alone is misleading. This is
why the success figure and the timing figure belong on the same page.

## Success rate

![PIBT success rate over agents and density for three solvers](../assets/experiments/pibt-success.png#only-light)
![PIBT success rate over agents and density for three solvers](../assets/experiments/pibt-success-dark.png#only-dark)

*Fraction of seeds returning a plan that passes cuplan's independent
validator. Rows are grids (32², 64²), columns are solvers, cells are
agents × obstacle density over seeds {0, 1, 2}. Timeouts count as
failures — the two at (64², 25%, 512) for pymapf. Every em dash in
this figure is cuplan CUDA at 64², lost to the device fault; PIBT
never triggered pymapf's timeout escalation, so no pymapf cell is
missing.*

**On the 32² grid the three solvers agree on 14 of 15 cells**,
non-monotone ones included: 67% at (32 agents, 5%), 100% at (32,
15%), 67% at (16, 25%), 33% at (128, 25%). PIBT's outcome is decided
by the inheritance chain, which is decided by the instance, so
independent implementations of the same rule fail the same draws.

The fifteenth cell is the interesting one. At (32², 15%, 128 agents)
pymapf solves all three seeds and both cuplan backends solve two.
That is a genuine difference, and it is a tie-break: when two moves
score equally, which one an agent takes decides whether the chain
that follows terminates. cuplan does not claim to dominate pymapf on
success rate, and here it does not.

The same asymmetry is larger on the 64² grid at high density. At
(64², 15%, 256 and 512 agents) pymapf reaches 67% where cuplan CPU
reaches 33%; at (64², 25%, 128) pymapf and cuplan both sit at 67%,
but at 256 agents cuplan gets one seed and pymapf gets none. Neither
ordering is stable — these are 3-seed cells, resolved to ±33% and no
finer, and reading a ranking out of them would be over-reading. What
is stable is the shape: **both libraries fall off the same cliff
between 128 and 512 agents once the grid is a quarter blocked.**

The honest reading of the timing table above is therefore
conditional. PIBT solves 512 agents on a 64² grid in 0.54 s *at 5%
density*, where every seed succeeds. At 25% density it solves none of
them.

## Solution quality

![Sum of costs and makespan of cuplan against pymapf on identical instances, plus CPU versus CUDA](../assets/experiments/quality-parity.png#only-light)
![Sum of costs and makespan of cuplan against pymapf on identical instances, plus CPU versus CUDA](../assets/experiments/quality-parity-dark.png#only-dark)

*Every instance solved by both solvers, both MAPF families pooled:
93 prioritized and 104 PIBT instances for the pymapf panels, 53 + 52
for the backend check. Grids 32²–128², all three densities, seeds
{0, 1, 2}. Diagonal = equal. Square markers are PIBT.*

On the 104 shared PIBT instances the median cuplan/pymapf
sum-of-costs ratio is **1.0000**, spread 0.882× to 1.126×; makespan
spreads wider, 0.807× to 1.171×. cuplan is cheaper on 39% of
instances and dearer on 48%, and only 13% match exactly — a much
looser cloud than prioritized planning's.

That looseness is expected and is not a defect. PIBT resolves ties
between equally-good moves, and a tie broken differently at timestep
3 sends two agents down different corridors for the rest of the run.
The plans diverge; the *cost* does not, systematically, in either
direction. What would be a defect is a cloud that sat off the
diagonal, and it does not.

The backend check is again exact: on all **52 of 52** shared
instances, cuplan's CPU and CUDA backends return identical sum of
costs and identical makespan. PIBT's inheritance chains run on the
host in both backends — only the distance oracle and the per-step
candidate ranking move to the device — so the two backends take the
same decisions in the same order by construction.

## What to take away

- PIBT is the family to reach for when you need a plan in
  milliseconds and can accept failure on crowded instances: 512
  agents on a 64² grid at 5% density, 0.54 s on the CPU backend.
- The GPU buys ~2× at 64², nothing reliable at 32², and the crossover
  is set by *grid area*, not agent count. Below 64², pass
  `backend="cpu"`.
- Quote PIBT timings with the density attached. At 25% on a 64² grid
  the same solver fails outright above 256 agents, and the timings of
  the runs that did succeed are not representative of the ones that
  did not.
