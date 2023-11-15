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

struct ChunkData
{
    Vertex* verts;
    GLuint* idx;
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
    glm::ivec2 windowSize;
    struct {
        glm::vec3 position;
        glm::vec3 forward;
        glm::vec3 up;
        glm::vec3 right;
        glm::vec2 pixelLength;
    } camera;

    struct {
        int frameId;
        uint32_t* colorBuffer;
        uint32_t* normalBuffer;
    } frame;
    OptixTraversableHandle rootHandle;
};

struct PRD {
    // Random random;
    glm::vec3  pixelColor;
    glm::vec3  pixelNormal;
    glm::vec3  pixelAlbedo;
};