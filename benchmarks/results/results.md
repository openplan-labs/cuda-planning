# Benchmark results

Machine: x86_64, GPU NVIDIA RTX A2000 Laptop GPU. Median over seeds; wall time
covers the full solve including host/device transfers. Random
grids at 15% obstacle density; identical instances for every
solver. Reproduce with `python -m cuplan.benchmark`.

| family | solver | grid | agents | success | median time (s) | median cost |
| :-- | :-- | --: | --: | --: | --: | --: |
| bfs | cuplan-cpu | 64x64 | 16 | 100% | 0.0092 | — |
| bfs | cuplan-cpu | 64x64 | 64 | 100% | 0.0371 | — |
| bfs | cuplan-cpu | 64x64 | 256 | 100% | 0.1508 | — |
| bfs | cuplan-cpu | 64x64 | 512 | 100% | 0.3328 | — |
| bfs | cuplan-cpu | 128x128 | 16 | 100% | 0.0504 | — |
| bfs | cuplan-cpu | 128x128 | 64 | 100% | 0.2168 | — |
| bfs | cuplan-cpu | 128x128 | 256 | 100% | 1.2521 | — |
| bfs | cuplan-cpu | 128x128 | 512 | 100% | 2.9883 | — |
| bfs | cuplan-cpu | 256x256 | 16 | 100% | 0.3706 | — |
| bfs | cuplan-cpu | 256x256 | 64 | 100% | 2.1415 | — |
| bfs | cuplan-cpu | 256x256 | 256 | 100% | 14.7985 | — |
| bfs | cuplan-cpu | 256x256 | 512 | 100% | 26.3726 | — |
| bfs | cuplan-cuda | 64x64 | 16 | 100% | 0.0034 | — |
| bfs | cuplan-cuda | 64x64 | 64 | 100% | 0.0055 | — |
| bfs | cuplan-cuda | 64x64 | 256 | 100% | 0.0130 | — |
| bfs | cuplan-cuda | 64x64 | 512 | 100% | 0.0233 | — |
| bfs | cuplan-cuda | 128x128 | 16 | 100% | 0.0104 | — |
| bfs | cuplan-cuda | 128x128 | 64 | 100% | 0.0237 | — |
| bfs | cuplan-cuda | 128x128 | 256 | 100% | 0.0815 | — |
| bfs | cuplan-cuda | 128x128 | 512 | 100% | 0.1569 | — |
| bfs | cuplan-cuda | 256x256 | 16 | 100% | 0.0388 | — |
| bfs | cuplan-cuda | 256x256 | 64 | 100% | 0.1512 | — |
| bfs | cuplan-cuda | 256x256 | 256 | 100% | 0.5894 | — |
| bfs | cuplan-cuda | 256x256 | 512 | 100% | 1.0251 | — |
| pibt | cuplan-cpu | 32x32 | 8 | 100% | 0.0050 | 149 |
| pibt | cuplan-cpu | 32x32 | 32 | 100% | 0.0117 | 764 |
| pibt | cuplan-cpu | 32x32 | 64 | 100% | 0.0196 | 1684 |
| pibt | cuplan-cpu | 32x32 | 128 | 67% | 0.0394 | 3990 |
| pibt | cuplan-cpu | 64x64 | 8 | 100% | 0.0114 | 326 |
| pibt | cuplan-cpu | 64x64 | 32 | 100% | 0.0299 | 1436 |
| pibt | cuplan-cpu | 64x64 | 64 | 100% | 0.0610 | 3043 |
| pibt | cuplan-cpu | 64x64 | 128 | 100% | 0.1199 | 7002 |
| pibt | cuplan-cuda | 32x32 | 8 | 100% | 0.0045 | 149 |
| pibt | cuplan-cuda | 32x32 | 32 | 100% | 0.0087 | 764 |
| pibt | cuplan-cuda | 32x32 | 64 | 100% | 0.0140 | 1684 |
| pibt | cuplan-cuda | 32x32 | 128 | 67% | 0.0277 | 3990 |
| pibt | cuplan-cuda | 64x64 | 8 | 100% | 0.0083 | 326 |
| pibt | cuplan-cuda | 64x64 | 32 | 100% | 0.0169 | 1436 |
| pibt | cuplan-cuda | 64x64 | 64 | 100% | 0.0262 | 3043 |
| pibt | cuplan-cuda | 64x64 | 128 | 100% | 0.0516 | 7002 |
| pibt | pymapf | 32x32 | 8 | 100% | 0.0200 | 149 |
| pibt | pymapf | 32x32 | 32 | 100% | 0.0856 | 757 |
| pibt | pymapf | 32x32 | 64 | 100% | 0.1544 | 1720 |
| pibt | pymapf | 32x32 | 128 | 100% | 0.3058 | 3929 |
| pibt | pymapf | 64x64 | 8 | 100% | 0.0688 | 326 |
| pibt | pymapf | 64x64 | 32 | 100% | 0.2816 | 1442 |
| pibt | pymapf | 64x64 | 64 | 100% | 0.5863 | 3006 |
| pibt | pymapf | 64x64 | 128 | 100% | 1.2626 | 7021 |
| prioritized | cuplan-cpu | 32x32 | 8 | 100% | 0.0141 | 149 |
| prioritized | cuplan-cpu | 32x32 | 32 | 100% | 0.0643 | 718 |
| prioritized | cuplan-cpu | 32x32 | 64 | 100% | 0.1271 | 1608 |
| prioritized | cuplan-cpu | 32x32 | 128 | 67% | 0.2487 | 3268 |
| prioritized | cuplan-cpu | 64x64 | 8 | 100% | 0.0341 | 326 |
| prioritized | cuplan-cpu | 64x64 | 32 | 100% | 0.1320 | 1344 |
| prioritized | cuplan-cpu | 64x64 | 64 | 100% | 0.2717 | 2748 |
| prioritized | cuplan-cpu | 64x64 | 128 | 100% | 0.6434 | 6275 |
| prioritized | cuplan-cuda | 32x32 | 8 | 100% | 0.0121 | 149 |
| prioritized | cuplan-cuda | 32x32 | 32 | 100% | 0.0445 | 718 |
| prioritized | cuplan-cuda | 32x32 | 64 | 100% | 0.0918 | 1608 |
| prioritized | cuplan-cuda | 32x32 | 128 | 67% | 0.1746 | 3268 |
| prioritized | cuplan-cuda | 64x64 | 8 | 100% | 0.0215 | 326 |
| prioritized | cuplan-cuda | 64x64 | 32 | 100% | 0.0682 | 1344 |
| prioritized | cuplan-cuda | 64x64 | 64 | 100% | 0.1330 | 2748 |
| prioritized | cuplan-cuda | 64x64 | 128 | 100% | 0.2911 | 6275 |
| prioritized | pymapf | 32x32 | 8 | 100% | 0.0122 | 149 |
| prioritized | pymapf | 32x32 | 32 | 100% | 0.2978 | 727 |
| prioritized | pymapf | 32x32 | 64 | 100% | 1.9881 | 1566 |
| prioritized | pymapf | 32x32 | 128 | 67% | 15.7464 | 3217 |
| prioritized | pymapf | 64x64 | 8 | 100% | 0.0447 | 326 |
| prioritized | pymapf | 64x64 | 32 | 100% | 1.2745 | 1358 |
| prioritized | pymapf | 64x64 | 64 | 100% | 6.8703 | 2780 |
| prioritized | pymapf | 64x64 | 128 | 100% | 46.3357 | 6194 |
| velocity_obstacles | cuplan-cpu | — | 16 | 100% | 0.0278 | — |
| velocity_obstacles | cuplan-cpu | — | 64 | 100% | 0.2863 | — |
| velocity_obstacles | cuplan-cpu | — | 128 | 100% | 1.3710 | — |
| velocity_obstacles | cuplan-cuda | — | 16 | 100% | 0.0429 | — |
| velocity_obstacles | cuplan-cuda | — | 64 | 100% | 0.1588 | — |
| velocity_obstacles | cuplan-cuda | — | 128 | 100% | 0.4429 | — |
