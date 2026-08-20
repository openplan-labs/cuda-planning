// Velocity-obstacle sampling — one thread per (agent, velocity sample).
//
// Mirrors pymapf's decentralized/velocity_obstacle semantics: for each
// neighbouring agent or moving obstacle, the collision cone is widened
// to 2.2 * radius, its two tangent lines are translated by the
// obstacle's velocity, and a candidate velocity is infeasible when it
// lies strictly inside both half-planes. Each thread scores one sample:
// +inf when infeasible, otherwise the distance to the desired velocity.
// The host does the per-agent argmin (a trivial reduction).
//
// others is (n_others, 4): x, y, vx, vy. For agent i, entry i is the
// agent itself and is skipped via the self_index array.

extern "C" __global__ void score_samples(
    const double* __restrict__ states,     // (n_agents, 4) x y vx vy
    const double* __restrict__ desired,    // (n_agents, 2)
    const double* __restrict__ others,     // (n_others, 4) x y vx vy
    const int* __restrict__ self_index,    // (n_agents,) index into others, or -1
    const double* __restrict__ samples,    // (n_samples, 2)
    double* __restrict__ scores,           // (n_agents, n_samples) out
    const int n_agents,
    const int n_others,
    const int n_samples,
    const double radius
) {
    const long long total = (long long)n_agents * n_samples;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int i = (int)(idx / n_samples);
        const int s = (int)(idx % n_samples);
        const double px = states[i * 4 + 0];
        const double py = states[i * 4 + 1];
        const double vx = samples[s * 2 + 0];
        const double vy = samples[s * 2 + 1];
        bool feasible = true;
        for (int j = 0; j < n_others && feasible; ++j) {
            if (j == self_index[i]) continue;
            const double ox = others[j * 4 + 0];
            const double oy = others[j * 4 + 1];
            const double ovx = others[j * 4 + 2];
            const double ovy = others[j * 4 + 3];
            const double dx = px - ox;
            const double dy = py - oy;
            double dist = sqrt(dx * dx + dy * dy);
            const double margin = 2.2 * radius;
            if (dist < margin) dist = margin;
            const double theta = atan2(dy, dx);
            const double half = asin(margin / dist);
            // Tangent line through the origin at angle phi has normal
            // n = (sin phi, -cos phi); translated by the obstacle
            // velocity t it becomes n . (v - t) = 0. The relative
            // velocity lies inside the cone (collision course) when
            // n_left . rv < 0 and n_right . rv > 0 — the sign
            // convention of pymapf's half-plane construction, verified
            // against the CPU reference in tests.
            const double rvx = vx - ovx;
            const double rvy = vy - ovy;
            const double phiL = theta + half;
            const double phiR = theta - half;
            const double leftSide = sin(phiL) * rvx - cos(phiL) * rvy;
            const double rightSide = sin(phiR) * rvx - cos(phiR) * rvy;
            if (leftSide < 0.0 && rightSide > 0.0) feasible = false;
        }
        if (!feasible) {
            scores[idx] = 1.0 / 0.0;  // +inf
        } else {
            const double ex = vx - desired[i * 2 + 0];
            const double ey = vy - desired[i * 2 + 1];
            scores[idx] = sqrt(ex * ex + ey * ey);
        }
    }
}
