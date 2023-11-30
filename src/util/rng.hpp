#pragma once

#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>
#include <glm/gtx/component_wise.hpp>

#pragma region utility functions

template<class T>
__device__ int manhattanLength(T v)
{
    return compAdd(abs(v));
}

template<class T>
__device__ int manhattanDistance(T a, T b)
{
    return compAdd(abs(a - b));
}

template<class T>
__host__ __device__ bool isInRange(T v, T min, T max)
{
    return v >= min && v <= max;
}

template<class T>
__device__ bool isPosInRange(T pos, T corner1, T corner2)
{
    T minPos = min(corner1, corner2);
    T maxPos = max(corner1, corner2);
    return pos.x >= minPos.x && pos.x <= maxPos.x
        && pos.y >= minPos.y && pos.y <= maxPos.y
        && pos.z >= minPos.z && pos.z <= maxPos.z;
}

__device__ float getRatio(float v, float minVal, float maxVal)
{
    return (v - minVal) / (maxVal - minVal);
}

__device__ float saturate(float v)
{
    return clamp(v, 0.f, 1.f);
}

__device__ float isSaturated(float v)
{
    return v >= 0.f && v <= 1.f;
}

__device__ bool calculateLineParams(const vec3 pos, const vec3 linePos1, const vec3 linePos2, float* ratio, float* distFromLine)
{
    vec3 vecLine = linePos2 - linePos1;

    vec3 pointPos = pos - linePos1;
    *ratio = dot(pointPos, vecLine) / dot(vecLine, vecLine);

    vec3 pointLine = vecLine * (*ratio);
    *distFromLine = distance(pointPos, pointLine);

    return isSaturated(*ratio);
}

#pragma endregion

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

__host__ __device__ vec3 rand3From3(vec3 v)
{
    return fract(sin(vec3(
        dot(v, vec3(238.68f, 491.28f, 402.98f)),
        dot(v, vec3(654.37f, 560.45f, 747.42f)),
        dot(v, vec3(640.88f, 151.81f, 674.81f))
    )) * 39021.426f);
}

#pragma endregion

#pragma region noise functions

__device__ vec2 simplex2From2(vec2 pos)
{
    return vec2(simplex(pos), simplex(pos + vec2(5923.45f, 4129.42f)));
}

template<int octaves = 5, class T>
__device__ float fbm(T pos)
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
__device__ vec2 fbm2From2(vec2 pos)
{
    return vec2(fbm<octaves>(pos), fbm<octaves>(pos + vec2(5923.45f, 4129.42f)));
}

template<int octaves = 5>
__device__ vec3 fbm3From3(vec3 pos)
{
    return vec3(fbm<octaves>(pos), fbm<octaves>(pos + vec3(5923.45f, 4129.42f, 5790.48f)), fbm<octaves>(pos + vec3(1765.68f, 4704.36f, 5692.12f)));
}

__device__ float worley(vec2 pos, vec3* colorPtr = nullptr, float* edgeDistPtr = nullptr)
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

    if (edgeDistPtr != nullptr)
    {
        *edgeDistPtr = (minDist2 - minDist1) * 0.5f;
    }

    return minDist1;
}

__device__ float worley(vec3 pos, vec3* colorPtr = nullptr, float* edgeDistPtr = nullptr)
{
    ivec3 uvInt = ivec3(floor(pos));
    vec3 uvFract = fract(pos);

    float minDist1 = FLT_MAX;
    float minDist2 = FLT_MAX;
    vec3 closestPoint;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                ivec3 neighbor = ivec3(x, y, z);
                vec3 point = rand3From3(uvInt + neighbor);
                vec3 diff = vec3(neighbor) + point - uvFract;
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
    }

    if (colorPtr != nullptr)
    {
        *colorPtr = rand3From3(closestPoint);
    }

    if (edgeDistPtr != nullptr)
    {
        *edgeDistPtr = (minDist2 - minDist1) * 0.5f;
    }

    return minDist1;
}

// idea comes from here: https://github.com/superfluke/WorleyCaves/blob/master/src/main/java/fluke/worleycaves/util/WorleyUtil.java#L315-L374
// not sure why minDist3 and minDist1 are swapped in the final return statement though
__device__ float specialCaveNoise(vec3 pos)
{
    ivec3 uvInt = ivec3(floor(pos));
    vec3 uvFract = fract(pos);

    float minDist1 = FLT_MAX;
    float minDist2 = FLT_MAX;
    float minDist3 = FLT_MAX;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                ivec3 neighbor = ivec3(x, y, z);
                vec3 point = rand3From3(uvInt + neighbor);
                vec3 diff = vec3(neighbor) + point - uvFract;
                float dist = length(diff);
                if (dist < minDist1)
                {
                    minDist3 = minDist2;
                    minDist2 = minDist1;
                    minDist1 = dist;
                }
                else if (dist < minDist2)
                {
                    minDist3 = minDist2;
                    minDist2 = dist;
                }
                else if (dist < minDist3)
                {
                    minDist3 = dist;
                }
            }
        }
    }

    return minDist3 / minDist1 - 1.f;
}

#pragma endregion