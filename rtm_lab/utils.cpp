// utils.cpp
#include "utils.h"
#include <iostream>
#include <cstdlib>

void checkCuda(cudaError_t result, const char* func)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << func
                  << " failed with " << cudaGetErrorString(result)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
