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