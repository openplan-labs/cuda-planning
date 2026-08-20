// Space-time wavefront expansion against a reservation table.
//
// One call advances the reachable set from timestep t to t+1 for a
// single agent: every free cell v checks its 5 candidate predecessors
// (4 orthogonal neighbours + itself for "wait"), honouring
//   - vertex reservations: v occupied at t+1,
//   - edge reservations: another agent moved v -> u arriving at t+1,
//     encoded as arrived_from[u] == v (only one agent can arrive at u).
//
// step_from records which move reached v first, for O(makespan) path
// reconstruction on the host. Move order matches cuplan.grid.MOVES:
// 0 up, 1 down, 2 left, 3 right, 4 wait.

extern "C" __global__ void spacetime_wave(
    const unsigned char* __restrict__ reach,      // (cells,) reachable at t
    unsigned char* __restrict__ reach_next,       // (cells,) out: reachable at t+1
    signed char* __restrict__ step_from,          // (cells,) out: move index or -1
    const unsigned char* __restrict__ free_mask,  // (cells,)
    const unsigned char* __restrict__ vertex_blocked, // (cells,) at t+1
    const int* __restrict__ arrived_from,         // (cells,) at t+1; -1 if none
    const int height,
    const int width,
    int* __restrict__ changed
) {
    const int cells = height * width;
    const int stride = blockDim.x * gridDim.x;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < cells; v += stride) {
        if (!free_mask[v] || vertex_blocked[v]) { reach_next[v] = 0; continue; }
        const int r = v / width;
        const int c = v % width;
        // Predecessor u for each move d satisfies u + d == v, so u = v - d.
        // moves: 0:(-1,0) 1:(1,0) 2:(0,-1) 3:(0,1) 4:(0,0)
        const int pred[5] = {
            (r + 1 < height) ? v + width : -1,  // arrived moving up: u below
            (r > 0)          ? v - width : -1,  // arrived moving down
            (c + 1 < width)  ? v + 1     : -1,  // arrived moving left
            (c > 0)          ? v - 1     : -1,  // arrived moving right
            v                                    // wait
        };
        signed char hit = -1;
        for (int d = 0; d < 5; ++d) {
            const int u = pred[d];
            if (u < 0 || !reach[u]) continue;
            // Edge (swap) check: someone arrives at u at t+1 coming from v.
            if (arrived_from[u] == v) continue;
            hit = (signed char)d;
            break;
        }
        if (hit >= 0) {
            reach_next[v] = 1;
            if (step_from[v] < 0) step_from[v] = hit;
            *changed = 1;
        } else {
            reach_next[v] = 0;
        }
    }
}
