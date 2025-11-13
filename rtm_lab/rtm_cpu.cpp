// rtm_cpu.cpp
#include "rtm_cpu.h"

#include <chrono>
#include <cmath>

void step_wavefield_2d_cpu(
    const float* p_prev,
    const float* p_curr,
    float*       p_next,
    const float* vel,
    const float* src,
    int Nx, int Nz,
    float dt, float h)
{
    float dt2 = dt * dt;
    float h2  = h * h;

    for (int j = 1; j < Nz - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            int idx       = j * Nx + i;
            int idx_right = j * Nx + (i + 1);
            int idx_left  = j * Nx + (i - 1);
            int idx_up    = (j - 1) * Nx + i;
            int idx_down  = (j + 1) * Nx + i;

            float p_c  = p_curr[idx];
            float p_r  = p_curr[idx_right];
            float p_l  = p_curr[idx_left];
            float p_u  = p_curr[idx_up];
            float p_d  = p_curr[idx_down];
            float p_m1 = p_prev[idx];

            float v    = vel[idx];
            float s    = src[idx];

            float lap = (p_r + p_l + p_d + p_u - 4.0f * p_c) / h2;

            p_next[idx] = 2.0f * p_c - p_m1 + dt2 * v * v * lap + dt2 * s;
        }
    }

    // Simple: copy boundaries from current (or keep zero)
    // Here we just leave them unchanged (whatever was in p_next).
}

void propagate_cpu(
    float* p0, float* p1, float* p2,
    const float* vel,
    const float* src_all,
    int Nx, int Nz, int Nt,
    float dt, float h,
    double& elapsed_seconds)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < Nt - 1; ++n) {
        const float* src_n = src_all + n * (Nx * Nz);

        step_wavefield_2d_cpu(
            p0, p1, p2,
            vel, src_n,
            Nx, Nz, dt, h);

        // Rotate pointers: p0 <- p1, p1 <- p2, p2 <- p0
        float* tmp = p0;
        p0 = p1;
        p1 = p2;
        p2 = tmp;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;
    elapsed_seconds = diff.count();
}
