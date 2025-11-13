// rtm_cpu.h
#ifndef RTM_CPU_H
#define RTM_CPU_H

void step_wavefield_2d_cpu(
    const float* p_prev,
    const float* p_curr,
    float*       p_next,
    const float* vel,
    const float* src,
    int Nx, int Nz,
    float dt, float h);

void propagate_cpu(
    float* p0, float* p1, float* p2,
    const float* vel,
    const float* src_all,
    int Nx, int Nz, int Nt,
    float dt, float h,
    double& elapsed_seconds);

#endif // RTM_CPU_H
