# Benchmark results

Machine: x86_64, GPU NVIDIA RTX A2000 Laptop GPU. Median over seeds; wall time
covers the full solve including host/device transfers. Random
grids at 15% obstacle density; identical instances for every
solver. Reproduce with `python -m cuplan.benchmark`.

| family | solver | grid | agents | success | median time (s) | median cost |
| :-- | :-- | --: | --: | --: | --: | --: |
| bfs | cuplan-cpu | 256x256 | 1024 | 100% | 54.2531 | — |
| bfs | cuplan-cuda | 256x256 | 1024 | 100% | 1.8572 | — |
| pibt | cuplan-cpu | 128x128 | 32 | 100% | 0.1235 | 3189 |
| pibt | cuplan-cpu | 128x128 | 64 | 100% | 0.2453 | 5154 |
| pibt | cuplan-cpu | 128x128 | 128 | 100% | 0.5034 | 12082 |
| pibt | cuplan-cpu | 128x128 | 256 | 100% | 1.1718 | 25879 |
| pibt | cuplan-cuda | 128x128 | 32 | 100% | 0.0406 | 3189 |
| pibt | cuplan-cuda | 128x128 | 64 | 100% | 0.0629 | 5154 |
| pibt | cuplan-cuda | 128x128 | 128 | 100% | 0.1191 | 12082 |
| pibt | cuplan-cuda | 128x128 | 256 | 100% | 0.2245 | 25879 |
| prioritized | cuplan-cpu | 128x128 | 32 | 100% | 0.5511 | 3083 |
| prioritized | cuplan-cpu | 128x128 | 64 | 100% | 0.8912 | 5085 |
| prioritized | cuplan-cpu | 128x128 | 128 | 100% | 2.0742 | 11798 |
| prioritized | cuplan-cpu | 128x128 | 256 | 100% | 4.5331 | 24231 |
| prioritized | cuplan-cuda | 128x128 | 32 | 100% | 0.1451 | 3083 |
| prioritized | cuplan-cuda | 128x128 | 64 | 100% | 0.2582 | 5085 |
| prioritized | cuplan-cuda | 128x128 | 128 | 100% | 0.5692 | 11798 |
| prioritized | cuplan-cuda | 128x128 | 256 | 100% | 1.1353 | 24231 |
| velocity_obstacles | cuplan-cpu | — | 256 | 100% | 5.5003 | — |
| velocity_obstacles | cuplan-cpu | — | 512 | 100% | 23.2283 | — |
| velocity_obstacles | cuplan-cuda | — | 256 | 100% | 1.4665 | — |
| velocity_obstacles | cuplan-cuda | — | 512 | 100% | 5.5590 | — |
