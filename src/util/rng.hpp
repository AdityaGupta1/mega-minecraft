#pragma once

#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>

#pragma region thrust rng

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

#pragma endregion

#pragma region random vector functions

__host__ __device__ float rand1From1(float v)
{
    return fract(sin(
        v * 238.68f
    ) * 39021.426f);
}

__host__ __device__ float rand1From2(vec2 v)
{
    return fract(sin(
        dot(v, vec2(238.68f, 491.28f))
    ) * 39021.426f);
}

__host__ __device__ float rand1From3(vec3 v)
{
    return fract(sin(
        dot(v, vec3(238.68f, 491.28f, 640.88f))
    ) * 39021.426f);
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

#pragma endregion

#pragma region noise functions

__device__ vec2 simplex2(vec2 pos)
{
    return vec2(simplex(pos), simplex(pos + vec2(5923.45f, 4129.42f)));
}

template<int octaves = 5>
__device__ float fbm(vec2 pos)
{
    float fbm = 0.f;
    float amplitude = 1.f;
#pragma unroll
    for (int i = 0; i < octaves; ++i)
    {
        amplitude *= 0.5f;
        fbm += amplitude * glm::simplex(pos);
        pos *= 2.f;
    }
    return fbm;
}

template<int octaves = 5>
__device__ vec2 fbm2(vec2 pos)
{
    return vec2(fbm<octaves>(pos), fbm<octaves>(pos + vec2(5923.45f, 4129.42f)));
}

__device__ float worley(vec2 pos, vec3* colorPtr = nullptr)
{
    ivec2 uvInt = ivec2(floor(pos));
    vec2 uvFract = fract(pos);

    float minDist = FLT_MAX;
    vec2 closestPoint;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            ivec2 neighbor = ivec2(x, y);
            vec2 point = rand2From2(uvInt + neighbor);
            vec2 diff = vec2(neighbor) + point - uvFract;
            float dist = length(diff);
            if (dist < minDist)
            {
                minDist = dist;
                closestPoint = point;
            }
        }
    }

    if (colorPtr != nullptr)
    {
        *colorPtr = rand3From2(closestPoint);
    }

    return minDist;
}

__device__ float worleyEdgeDist(vec2 pos, vec3* colorPtr = nullptr)
{
    ivec2 uvInt = ivec2(floor(pos));
    vec2 uvFract = fract(pos);

    float minDist1 = FLT_MAX;
    float minDist2 = FLT_MAX;
    vec2 closestPoint;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            ivec2 neighbor = ivec2(x, y);
            vec2 point = rand2From2(uvInt + neighbor);
            vec2 diff = vec2(neighbor) + point - uvFract;
            float dist = length(diff);
            if (dist < minDist1)
            {
                minDist2 = minDist1;
                minDist1 = dist;
                closestPoint = point;
            }
            else if (dist < minDist2)
            {
                minDist2 = dist;
            }
        }
    }

    if (colorPtr != nullptr)
    {
        *colorPtr = rand3From2(closestPoint);
    }

    return (minDist2 - minDist1) * 0.5f;
}

#pragma endregion