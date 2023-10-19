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