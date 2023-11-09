#include "cudaUtils.hpp"

#include <iostream>

void CudaUtils::checkCUDAError(const char* msg, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        if (line >= 0)
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void CudaUtils::cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA call (%s) failed with error: '%s' (%s:%u)\n", call, cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}