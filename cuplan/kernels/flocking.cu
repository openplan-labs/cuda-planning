// Boids force accumulation — one thread per agent.
//
// Mirrors pymapf.swarm.flocking.Boids (Reynolds 1987): within the
// perception radius, separation is an inverse-square repulsion below
// the separation distance, cohesion steers toward the mean neighbour
// offset, alignment toward the mean neighbour velocity. The command is
// clamped to max_accel; the host integrator clamps speed.
//
// The O(n^2) neighbour scan is deliberate: at the swarm sizes this
// library targets (<= a few thousand agents) a brute-force scan on the
// device beats building spatial structures every step.

extern "C" __global__ void boids_forces(
    const double* __restrict__ positions,   // (n, dim)
    const double* __restrict__ velocities,  // (n, dim)
    double* __restrict__ commands,          // (n, dim) out
    const int n,
    const int dim,
    const double perception_radius,
    const double separation_distance,
    const double separation_gain,
    const double cohesion_gain,
    const double alignment_gain,
    const double max_accel
) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        double sep[3] = {0.0, 0.0, 0.0};
        double coh[3] = {0.0, 0.0, 0.0};
        double ali[3] = {0.0, 0.0, 0.0};
        int neighbours = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            double off[3];
            double sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                off[d] = positions[j * dim + d] - positions[i * dim + d];
                sq += off[d] * off[d];
            }
            if (sq > perception_radius * perception_radius) continue;
            ++neighbours;
            double dist = sqrt(sq);
            if (dist < 1e-6) dist = 1e-6;
            if (dist < separation_distance) {
                for (int d = 0; d < dim; ++d)
                    sep[d] -= off[d] / (dist * dist);
            }
            for (int d = 0; d < dim; ++d) {
                coh[d] += off[d];
                ali[d] += velocities[j * dim + d] - velocities[i * dim + d];
            }
        }
        double cmd[3] = {0.0, 0.0, 0.0};
        if (neighbours > 0) {
            for (int d = 0; d < dim; ++d) {
                cmd[d] = separation_gain * sep[d]
                       + cohesion_gain * (coh[d] / neighbours)
                       + alignment_gain * (ali[d] / neighbours);
            }
        }
        double norm = 0.0;
        for (int d = 0; d < dim; ++d) norm += cmd[d] * cmd[d];
        norm = sqrt(norm);
        if (norm > max_accel && norm > 0.0) {
            for (int d = 0; d < dim; ++d) cmd[d] *= max_accel / norm;
        }
        for (int d = 0; d < dim; ++d) commands[i * dim + d] = cmd[d];
    }
}
