#pragma once

#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector_math.h"

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
    float3 pos;
    float3 nor; // TODO: compact this
    float2 uv;
    Mats m{ Mats::M_DIFFUSE };
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

    float time;

    float3 sunDir;
    float3 moonDir;

    float3 starsRotateMatX; // columns of a 3x3 matrix, inverse of sunRotateMat
    float3 starsRotateMatY;
    float3 starsRotateMatZ;
};

struct ChunkData
{
    Vertex* verts;
    uint3* idx;
    cudaTextureObject_t tex_diffuse;
    cudaTextureObject_t tex_emissive;
    cudaTextureObject_t tex_normal;
};

struct Texture
{
    uint8_t* host_buffer;
    int32_t width;
    int32_t height;
    int32_t channels;
};

struct PRD {
    uint32_t seed;

    bool isDone;
    bool needsFirstHitData;
    bool specularHit;

    bool foundLightSource;
    bool scattered;

    struct
    {
        float3 pos;
        float3 newDir;
    } isect;

    float3  rayColor;
    float3  pixelColor;
    float3  pixelAlbedo;
    float3  pixelNormal;
    float3 fogColor;
    float3 scatterPosition;
    float fogFactor;
    float scatterFactor;
};