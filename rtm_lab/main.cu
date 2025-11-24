// main.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include "rtm_cpu.h"
#include "utils.h"

constexpr int NX = 2000;
constexpr int NZ = 2000;
constexpr int NT = 500;

constexpr float DT = 0.001f;   // time step
constexpr float H  = 10.0f;    // spatial step (dx = dz = H)
constexpr float V0 = 1500.0f;  // background velocity (m/s)

// PART A1: Baseline Global Memory Kernel
__global__
void step_wavefield_2d_gpu(
    const float* __restrict__ p_prev,
    const float* __restrict__ p_curr,
    float* __restrict__ p_next,
    const float* __restrict__ vel,
    const float* __restrict__ src,
    int Nx, int Nz,
    float dt, float h)
{
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: strictly interior points (1 to N-2)
    // The stencil requires neighbors i+/-1 and j+/-1
    if (i < 1 || i >= Nx - 1 || j < 1 || j >= Nz - 1) {
        return;
    }

    // Flattened indices
    int idx = j * Nx + i;
    
    // Neighbor indices
    int idx_right = idx + 1;
    int idx_left  = idx - 1;
    int idx_down  = (j + 1) * Nx + i; // j+1 is down in memory
    int idx_up    = (j - 1) * Nx + i;

    // Precompute constants
    float h2 = h * h;
    float dt2 = dt * dt;

    // Data from Global Memory
    float pc = p_curr[idx];
    float pr = p_curr[idx_right];
    float pl = p_curr[idx_left];
    float pd = p_curr[idx_down];
    float pu = p_curr[idx_up];
    
    float pp = p_prev[idx];
    float v  = vel[idx];
    float s  = src[idx];

    // Compute Laplacian (5-point stencil)
    float lap = (pr + pl + pd + pu - 4.0f * pc) / h2;

    // Time update
    p_next[idx] = 2.0f * pc - pp + (dt2 * v * v * lap) + (dt2 * s);
}

// PART D: Shared Memory Tiled Kernel
template<int BLOCK_X, int BLOCK_Z>
__global__
void step_wavefield_2d_tiled(
    const float* __restrict__ p_prev,
    const float* __restrict__ p_curr,
    float* __restrict__ p_next,
    const float* __restrict__ vel,
    const float* __restrict__ src,
    int Nx, int Nz,
    float dt, float h)
{
    // Shared memory tile size: Block dimensions + 2 for Halo (apron)
    __shared__ float s_p[BLOCK_Z + 2][BLOCK_X + 2];

    // Local thread indices
    int tx = threadIdx.x ; // TODO add +1 instead of it layter
    int ty = threadIdx.y;

    // Global coordinates
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    
    int idx = j * Nx + i;

    // 1. Load Interior: Map thread to s_p[ty+1][tx+1]
    // We guard load against global image boundaries
    if (i < Nx && j < Nz) {
        s_p[ty + 1][tx + 1] = p_curr[idx];
    }

    // 2. Load Halos (Ghost Cells)
    // We only load if the neighbor exists in global memory
    
    // Top Halo (ty == 0)
    if (ty == 0) {
        int global_j_up = j - 1;
        if (global_j_up >= 0 && i < Nx) 
            s_p[0][tx + 1] = p_curr[idx - Nx];
    }
    // Bottom Halo (ty == last)
    if (ty == BLOCK_Z - 1) {
        int global_j_down = j + 1;
        if (global_j_down < Nz && i < Nx) 
            s_p[BLOCK_Z + 1][tx + 1] = p_curr[idx + Nx];
    }
    // Left Halo (tx == 0)
    if (tx == 0) {
        int global_i_left = i - 1;
        if (global_i_left >= 0 && j < Nz) 
            s_p[ty + 1][0] = p_curr[idx - 1];
    }
    // Right Halo (tx == last)
    if (tx == BLOCK_X - 1) {
        int global_i_right = i + 1;
        if (global_i_right < Nx && j < Nz) 
            s_p[ty + 1][BLOCK_X + 1] = p_curr[idx + 1];
    }

    __syncthreads();


    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Nz - 1) {
        float h2 = h * h;
        float dt2 = dt * dt;

        // Read neighbors from Shared Memory
        float pc = s_p[ty + 1][tx + 1];
        float pr = s_p[ty + 1][tx + 2]; // Right
        float pl = s_p[ty + 1][tx + 0]; // Left
        float pd = s_p[ty + 2][tx + 1]; // Down
        float pu = s_p[ty + 0][tx + 1]; // Up
        
        // Read other arrays from Global (Point-wise, no neighbors needed)
        float pp = p_prev[idx];
        float v  = vel[idx];
        float s  = src[idx];

        float lap = (pr + pl + pd + pu - 4.0f * pc) / h2;
        p_next[idx] = 2.0f * pc - pp + (dt2 * v * v * lap) + (dt2 * s);
    }
}


void propagate_gpu_baseline(
    float* d_p0, float* d_p1, float* d_p2,
    const float* d_vel, const float* d_src_all,
    int Nx, int Nz, int Nt,
    float dt, float h,
    float& elapsed_ms)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((Nx + blockDim.x - 1) / blockDim.x,
                 (Nz + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // PART A2: Time Stepping Loop
    for (int n = 0; n < Nt - 1; ++n) {
        const float* d_src_n = d_src_all + n * (Nx * Nz);

        // Launch Kernel
        step_wavefield_2d_gpu<<<gridDim, blockDim>>>(
            d_p0,   // prev
            d_p1,   // curr
            d_p2,   // next
            d_vel, d_src_n,
            Nx, Nz, dt, h
        );

        // Pointer Rotation: p_prev <- p_curr, p_curr <- p_next, p_next <- p_prev (buffer reuse)
        float* temp = d_p0;
        d_p0 = d_p1;
        d_p1 = d_p2;
        d_p2 = temp;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}


void propagate_gpu_tiled(
    float* d_p0, float* d_p1, float* d_p2,
    const float* d_vel, const float* d_src_all,
    int Nx, int Nz, int Nt,
    float dt, float h,
    float& elapsed_ms)
{
    // Match the template arguments
    const int BX = 16;
    const int BZ = 16;
    dim3 blockDim(BX, BZ);
    dim3 gridDim((Nx + blockDim.x - 1) / blockDim.x,
                 (Nz + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int n = 0; n < Nt - 1; ++n) {
        const float* d_src_n = d_src_all + n * (Nx * Nz);

        step_wavefield_2d_tiled<BX, BZ><<<gridDim, blockDim>>>(
            d_p0, d_p1, d_p2,
            d_vel, d_src_n,
            Nx, Nz, dt, h
        );

        float* temp = d_p0;
        d_p0 = d_p1;
        d_p1 = d_p2;
        d_p2 = temp;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}


void initialize_velocity(std::vector<float>& vel, int Nx, int Nz) {
    std::fill(vel.begin(), vel.end(), V0);
}

void initialize_source(std::vector<float>& src_all, int Nx, int Nz, int Nt) {
    int ix0 = Nx / 2;
    int iz0 = Nz / 2;
    int idx0 = iz0 * Nx + ix0;
    float f0 = 15.0f; 
    for (int n = 0; n < Nt; ++n) {
        float t = n * DT;
        float pi_f0_t = static_cast<float>(M_PI) * f0 * (t - 1.0f / f0);
        float ricker  = (1.0f - 2.0f * pi_f0_t * pi_f0_t) * std::exp(-pi_f0_t * pi_f0_t);
        src_all[n * Nx * Nz + idx0] = ricker;
    }
}

int main()
{
    const int Nx = NX;
    const int Nz = NZ;
    const int Nt = NT;
    const int N  = Nx * Nz;

    std::cout << "RTM 2D stencil lab solution\n";
    std::cout << "Grid: " << Nx << " x " << Nz << ", Nt = " << Nt << std::endl;

    // Host arrays
    std::vector<float> h_p0(N, 0.0f);
    std::vector<float> h_p1(N, 0.0f);
    std::vector<float> h_p2(N, 0.0f);
    std::vector<float> h_vel(N, 0.0f);
    std::vector<float> h_src_all(static_cast<size_t>(Nt) * N, 0.0f);

    initialize_velocity(h_vel, Nx, Nz);
    initialize_source(h_src_all, Nx, Nz, Nt);

    // 1. CPU Reference
    std::cout << "Running CPU reference...\n";
    std::vector<float> h_p0_cpu = h_p0;
    std::vector<float> h_p1_cpu = h_p1;
    std::vector<float> h_p2_cpu = h_p2;
    double cpu_time_s = 0.0;
    propagate_cpu(h_p0_cpu.data(), h_p1_cpu.data(), h_p2_cpu.data(),
                  h_vel.data(), h_src_all.data(),
                  Nx, Nz, Nt, DT, H, cpu_time_s);
    std::cout << "CPU time: " << cpu_time_s << " s\n";

    // 2. GPU Setup
    float *d_p0, *d_p1, *d_p2, *d_vel, *d_src_all;
    CHECK_CUDA(cudaMalloc(&d_p0, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p1, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p2, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_src_all, static_cast<size_t>(Nt) * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_p0, h_p0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p1, h_p1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p2, h_p2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel, h_vel.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src_all, h_src_all.data(), static_cast<size_t>(Nt) * N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Run Baseline GPU
    std::cout << "Running Baseline GPU...\n";
    float gpu_ms = 0.0f;
    propagate_gpu_baseline(d_p0, d_p1, d_p2, d_vel, d_src_all, Nx, Nz, Nt, DT, H, gpu_ms);

    // Verify Baseline
    std::vector<float> h_p_curr_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_p_curr_gpu.data(), d_p1, N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff_base = 0.0f;
    for (int i = 0; i < N; ++i) max_diff_base = std::max(max_diff_base, std::fabs(h_p1_cpu[i] - h_p_curr_gpu[i]));

    // 4. Run Tiled GPU
    // Reset device memory to 0 for fair test
    CHECK_CUDA(cudaMemcpy(d_p0, h_p0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p1, h_p1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p2, h_p2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Running Tiled GPU...\n";
    float tiled_ms = 0.0f;
    propagate_gpu_tiled(d_p0, d_p1, d_p2, d_vel, d_src_all, Nx, Nz, Nt, DT, H, tiled_ms);

    // Verify Tiled
    CHECK_CUDA(cudaMemcpy(h_p_curr_gpu.data(), d_p1, N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff_tiled = 0.0f;
    for (int i = 0; i < N; ++i) max_diff_tiled = std::max(max_diff_tiled, std::fabs(h_p1_cpu[i] - h_p_curr_gpu[i]));

    // Reporting...
    std::cout << "------------------------------------------------\n";
    std::cout << "Performance Report:\n";
    std::cout << "CPU Time:      " << cpu_time_s << " s\n";
    std::cout << "Baseline GPU:  " << gpu_ms / 1000.0f << " s | Speedup: " << cpu_time_s / (gpu_ms/1000.0f) << "x\n";
    std::cout << "Tiled GPU:     " << tiled_ms / 1000.0f << " s | Speedup: " << cpu_time_s / (tiled_ms/1000.0f) << "x\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "Correctness (Max Abs Error):\n";
    std::cout << "Baseline: " << max_diff_base << "\n";
    std::cout << "Tiled:    " << max_diff_tiled << "\n";

    CHECK_CUDA(cudaFree(d_p0));
    CHECK_CUDA(cudaFree(d_p1));
    CHECK_CUDA(cudaFree(d_p2));
    CHECK_CUDA(cudaFree(d_vel));
    CHECK_CUDA(cudaFree(d_src_all));

    return 0;
}