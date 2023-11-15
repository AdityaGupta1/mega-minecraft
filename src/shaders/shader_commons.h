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

struct ChunkData
{
    Vertex* verts;
    uint32_t* idx;
    cudaTextureObject_t texture;
};

struct Texture
{
    uint8_t* host_buffer;
    int32_t width;
    int32_t height;
    int32_t channels;
};

struct OptixParams
{
    int numPixelSamples = 1;
    int2 windowSize;
    struct {
        float3 position;
        float3 forward;
        float3 up;
        float3 right;
        float2 pixelLength;
    } camera;

    struct {
        int frameId;
        float4* colorBuffer;
        float4* normalBuffer;
    } frame;
    OptixTraversableHandle rootHandle;
};

struct PRD {
    // Random random;
    float3  pixelColor;
    float3  pixelNormal;
    float3  pixelAlbedo;
};