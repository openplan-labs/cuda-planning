// Batched grid BFS (flood fill) — one distance map per source.
//
// Gather ("pull") formulation: at wave t, every unlabelled free cell
// checks whether any 4-neighbour was labelled t-1. Pull avoids atomics
// entirely and keeps memory access coalesced along rows; the cost is
// touching settled cells each wave, which the batch over maps amortises.
//
// Layout: dist is (n_maps, height * width) int32, -1 meaning unreached.
// free_mask is (height * width) uint8 shared by every map in the batch.

extern "C" __global__ void bfs_wave(
    int* __restrict__ dist,
    const unsigned char* __restrict__ free_mask,
    const int n_maps,
    const int height,
    const int width,
    const int wave,          // distance label being assigned this call
    int* __restrict__ changed // set to 1 when any cell was labelled
) {
    const int cells = height * width;
    const long long total = (long long)n_maps * cells;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int map = (int)(idx / cells);
        const int cell = (int)(idx % cells);
        if (!free_mask[cell]) continue;
        int* d = dist + (long long)map * cells;
        if (d[cell] != -1) continue;
        const int r = cell / width;
        const int c = cell % width;
        const int prev = wave - 1;
        bool touched =
            (r > 0          && d[cell - width] == prev) ||
            (r + 1 < height && d[cell + width] == prev) ||
            (c > 0          && d[cell - 1]     == prev) ||
            (c + 1 < width  && d[cell + 1]     == prev);
        if (touched) {
            d[cell] = wave;
            *changed = 1;
        }
    }
}
