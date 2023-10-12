#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"

#include <iostream>

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos), heightfield(), blocks()
{
}

void Chunk::dummyFill()
{
    for (int y = 0; y < 64; ++y)
    {
        for (int z = 0; z < 16; ++z)
        {
            for (int x = 0; x < 16; ++x)
            {
                this->blocks[x + 16 * z + 256 * y] = Block::STONE;
            }
        }
    }
}

void Chunk::createVBOs()
{
    GLuint idx[3]{ 0, 1, 2 };
    Vertex verts[3]{
        { glm::vec3(-1.f, -1.f, 0.9999f) },
        { glm::vec3(1.f, -1.f, 0.9999f) },
        { glm::vec3(1.f, 1.f, 0.9999f) }
    };

    idxCount = 3;

    generateIdx();
    bindIdx();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), idx, GL_STATIC_DRAW); // TODO helper functions for this and next glBufferData call in Drawable

    generateVerts();
    bindVerts();
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(Vertex), verts, GL_STATIC_DRAW);
}