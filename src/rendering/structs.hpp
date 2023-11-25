#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <cuda_runtime.h>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor; // TODO: compact this
    glm::vec2 uv;
};

struct OptixParams
{
    OptixTraversableHandle rootHandle;

    struct
    {
        float4* colorBuffer;
        float4* albedoBuffer;
        float4* normalBuffer;
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

    float3 sunDir;
};

struct ChunkData
{
    Vertex* verts;
    glm::uvec3* idx;
    cudaTextureObject_t tex_diffuse;
    cudaTextureObject_t tex_emissive;
};