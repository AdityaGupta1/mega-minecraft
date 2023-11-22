#pragma once

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)							                    \
  {								                                     	\
     cudaError_t err = call;                                            \
     if (err != cudaSuccess) {                                          \
       fprintf(stderr, "CUDA call (%s) failed with error: '%s (%s)' (%s:%u)\n", #call, cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
       exit(EXIT_FAILURE);                                              \
     }                                                                  \
  }

#define CU_CHECK(call)                                                  \
{                                                                       \
  CUresult err = call;                                                  \
  if (err != CUDA_SUCCESS) {                                            \
    const char *errName, *errStr;                                       \
    cuGetErrorName(err, &errName);                                      \
    cuGetErrorString(err, &errStr);                                     \
    fprintf(stderr, "CUDA call (%s) failed with error: '%s (%s)' (%s:%u)\n", #call, errName, errStr, __FILE__, __LINE__);\
  }                                                                     \
}

#define OPTIX_CHECK( call )                                                                          \
  {                                                                                                  \
    OptixResult res = call;                                                                          \
    if( res != OPTIX_SUCCESS )                                                                       \
      {                                                                                              \
        fprintf( stderr, "Optix call (%s) failed with error: '%s (%s)' (%s:%u)\n", #call, optixGetErrorName(res), optixGetErrorString(res), __FILE__, __LINE__ ); \
      }                                                                                              \
  }

// exit(EXIT_FAILURE);                                                                          \