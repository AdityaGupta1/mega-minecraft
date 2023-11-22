#pragma once

#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector_math.h"

struct Vertex
{
    float3 pos;
    float3 nor; // TODO: compact this
    float2 uv;
};

struct OptixParams
{
    OptixTraversableHandle rootHandle;

    struct
    {
        uint32_t* colorBuffer;
        uint32_t* normalBuffer;
        int frameId;
    } frame;

    int numPixelSamples = 1;
    int2 windowSize;

    struct
    {
        float3 position;
        float3 forward;
        float3 up;
        float3 right;
        float2 pixelLength;
    } camera;
};

struct ChunkData
{
    Vertex* verts;
    uint3* idx;
};

struct Texture
{
    uint8_t* host_buffer;
    int32_t width;
    int32_t height;
    int32_t channels;
};

struct PRD {
    // Random random;
    float3  pixelColor;
    float3  pixelNormal;
    float3  pixelAlbedo;
};