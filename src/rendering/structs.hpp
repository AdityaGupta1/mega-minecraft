#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <cuda_runtime.h>

enum class Mats : size_t {
    M_DIFFUSE,
    M_WATER,
    M_CRYSTAL,
    M_SMOOTH_MICRO,
    M_MICRO,
    M_ROUGH_MICRO
};

struct Mat
{
    float ior;
    float roughness;
    bool reflecting;
    bool refracting;
    bool wavy;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor; // TODO: compact this
    glm::vec2 uv;
    Mats m {Mats::M_DIFFUSE};
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
    cudaTextureObject_t tex_normal;
};