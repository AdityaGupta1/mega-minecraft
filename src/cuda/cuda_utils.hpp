#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

namespace CudaUtils
{
    void checkCUDAError(const char* msg, int line = -1);
}