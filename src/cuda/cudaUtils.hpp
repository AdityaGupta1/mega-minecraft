#pragma once

#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/random.h>
#include <vector>
#include "util/utils.hpp"
#include "util/common.h"


namespace CudaUtils
{

    void checkCUDAError(const char* msg, int line = -1);

}

// Adapted from https://github.com/ingowald/optix7course/blob/master/example12_denoiseSeparateChannels/CUDABuffer.h
struct CUBuffer {
    void* d_ptr{ nullptr };
    size_t byteSize{ 0 };

    inline CUdeviceptr dev_ptr() const {
        return (CUdeviceptr)d_ptr;
    }

    inline size_t size() const {
        return byteSize;
    }

    void alloc(size_t size) {
        assert(d_ptr == nullptr);
        this->byteSize = size;
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, byteSize));
    }

    void free() {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        byteSize = 0;
    }

    void resize(size_t size) {
        if (d_ptr)
            free();
        alloc(size);
    }

    template<typename T>
    void initFromVector(const std::vector<T>& v) {
        alloc(v.size() * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, v.data(), byteSize, cudaMemcpyHostToDevice));
    }

    template<typename T>
    void populate(const T* host_ptr, size_t count) {
        assert(d_ptr != nullptr);
        assert(byteSize >= count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void retrieve(T* host_ptr, size_t count) {
        assert(d_ptr != nullptr);
        assert(byteSize >= count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};