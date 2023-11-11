#pragma once

#include "cuda/cudaUtils.hpp"

__host__ __device__ inline unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x)
{
    int h = hash(x);
    return thrust::default_random_engine(h);
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x, int y, int z)
{
    int h = hash((1 << 31) | (x << 22) | y) ^ hash(z);
    return thrust::default_random_engine(h);
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x, int y, int z, int w)
{
    int h = hash((1 << 31) | (x << 22) | (y << 11) | w) ^ hash(z);
    return thrust::default_random_engine(h);
}

__host__ __device__ vec2 rand2From2(vec2 v)
{
    return fract(sin(vec2(
        dot(v, vec2(238.68f, 491.28f)),
        dot(v, vec2(654.37f, 560.45f))
    )) * 39021.426f);
}

__host__ __device__ vec2 rand2From3(vec3 v)
{
    return fract(sin(vec2(
        dot(v, vec3(238.68f, 491.28f, 640.88f)),
        dot(v, vec3(654.37f, 560.45f, 151.81f))
    )) * 39021.426f);
}

__host__ __device__ vec3 rand3From2(vec2 v)
{
    return fract(sin(vec3(
        dot(v, vec2(238.68f, 491.28f)),
        dot(v, vec2(654.37f, 560.45f)),
        dot(v, vec2(640.88f, 151.81f))
    )) * 39021.426f);
}