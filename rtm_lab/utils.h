// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* func);
#define CHECK_CUDA(call) checkCuda((call), #call)

#endif // UTILS_H
