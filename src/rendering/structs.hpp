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
    glm::uvec3* idx;
    cudaTextureObject_t tex_diffuse;
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
    glm::vec3  pixelColor;
    glm::vec3  pixelNormal;
    glm::vec3  pixelAlbedo;
};