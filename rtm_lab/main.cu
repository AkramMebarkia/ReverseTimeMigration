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

__global__
void step_wavefield_2d_gpu(
    const float* __restrict__ p_prev,
    const float* __restrict__ p_curr,
    float*       __restrict__ p_next,
    const float* __restrict__ vel,
    const float* __restrict__ src,
    int Nx, int Nz,
    float dt, float h)
{
    //  - Compute i, j from blockIdx/threadIdx
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO [Lab Part A1]:
    //  - Skip boundaries
    //  - Compute flattened indices (idx, neighbors)
    //  - Load neighbors, compute Laplacian
    //  - Write p_next[idx]

    // Stub so the template compiles (does nothing)
    (void)p_prev; (void)p_curr; (void)p_next;
    (void)vel; (void)src; (void)Nx; (void)Nz;
    (void)dt; (void)h;
}

template<int BLOCK_X, int BLOCK_Z>
__global__
void step_wavefield_2d_tiled(
    const float* __restrict__ p_prev,
    const float* __restrict__ p_curr,
    float*       __restrict__ p_next,
    const float* __restrict__ vel,
    const float* __restrict__ src,
    int Nx, int Nz,
    float dt, float h)
{
    // TODO [Lab Part D]:
    //  - Declare shared memory tile with halo
    //  - Load interior points and halos
    //  - __syncthreads()
    //  - Use shared memory to compute Laplacian
    //  - Write p_next[idx]

    (void)p_prev; (void)p_curr; (void)p_next;
    (void)vel; (void)src; (void)Nx; (void)Nz;
    (void)dt; (void)h;
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

    for (int n = 0; n < Nt - 1; ++n) {
        const float* d_src_n = d_src_all + n * (Nx * Nz);

        // TODO [Lab Part A2]:
        //  - Launch step_wavefield_2d_gpu<<<gridDim, blockDim>>>(...)
        //  - Rotate pointers (d_p0, d_p1, d_p2)

        (void)d_src_n; // silence unused warning in template
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    elapsed_ms = ms;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

void initialize_velocity(std::vector<float>& vel, int Nx, int Nz)
{
    std::fill(vel.begin(), vel.end(), V0);
}

void initialize_source(std::vector<float>& src_all, int Nx, int Nz, int Nt)
{
    // Simple Ricker-like pulse at grid center, injected at a few time steps
    int ix0 = Nx / 2;
    int iz0 = Nz / 2;
    int idx0 = iz0 * Nx + ix0;

    float f0 = 15.0f; // dominant frequency [Hz]
    for (int n = 0; n < Nt; ++n) {
        float t = n * DT;
        float pi_f0_t = static_cast<float>(M_PI) * f0 * (t - 1.0f / f0);
        float ricker  = (1.0f - 2.0f * pi_f0_t * pi_f0_t)
                        * std::exp(-pi_f0_t * pi_f0_t);
        src_all[n * Nx * Nz + idx0] = ricker;
    }
}

int main()
{
    const int Nx = NX;
    const int Nz = NZ;
    const int Nt = NT;
    const int N  = Nx * Nz;

    std::cout << "RTM 2D stencil lab template\n";
    std::cout << "Grid: " << Nx << " x " << Nz
              << ", Nt = " << Nt << std::endl;

    // Host arrays
    std::vector<float> h_p0(N, 0.0f);
    std::vector<float> h_p1(N, 0.0f);
    std::vector<float> h_p2(N, 0.0f);
    std::vector<float> h_vel(N, 0.0f);
    std::vector<float> h_src_all(static_cast<size_t>(Nt) * N, 0.0f);

    initialize_velocity(h_vel, Nx, Nz);
    initialize_source(h_src_all, Nx, Nz, Nt);

    // CPU reference
    std::vector<float> h_p0_cpu = h_p0;
    std::vector<float> h_p1_cpu = h_p1;
    std::vector<float> h_p2_cpu = h_p2;

    double cpu_time_s = 0.0;
    propagate_cpu(
        h_p0_cpu.data(), h_p1_cpu.data(), h_p2_cpu.data(),
        h_vel.data(), h_src_all.data(),
        Nx, Nz, Nt, DT, H, cpu_time_s);

    const float* h_p_curr_cpu = h_p1_cpu.data();

    std::cout << "CPU propagation time: " << cpu_time_s << " s\n";

    // Device arrays
    float *d_p0 = nullptr, *d_p1 = nullptr, *d_p2 = nullptr;
    float *d_vel = nullptr, *d_src_all = nullptr;

    CHECK_CUDA(cudaMalloc(&d_p0, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p1, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p2, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_src_all, static_cast<size_t>(Nt) * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_p0, h_p0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p1, h_p1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_p2, h_p2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel, h_vel.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src_all, h_src_all.data(),
                          static_cast<size_t>(Nt) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    float gpu_ms = 0.0f;
    propagate_gpu_baseline(
        d_p0, d_p1, d_p2,
        d_vel, d_src_all,
        Nx, Nz, Nt, DT, H,
        gpu_ms);

    // Copy back current wavefield (p1) for comparison
    std::vector<float> h_p_curr_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_p_curr_gpu.data(), d_p1,
                          N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU vs GPU (will not match until students implement kernels)
    float max_abs_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(h_p_curr_cpu[i] - h_p_curr_gpu[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    std::cout << "GPU baseline propagation time: " << gpu_ms / 1000.0f << " s\n";
    std::cout << "Max abs difference (CPU vs GPU) = " << max_abs_diff << std::endl;
    std::cout << "(Large difference is expected until GPU kernel is implemented.)\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_p0));
    CHECK_CUDA(cudaFree(d_p1));
    CHECK_CUDA(cudaFree(d_p2));
    CHECK_CUDA(cudaFree(d_vel));
    CHECK_CUDA(cudaFree(d_src_all));

    return 0;
}
