#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"

#include <iostream>

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos)
{
    std::fill_n(heightfield.begin(), 256, 0);
    std::fill_n(blocks.begin(), 65536, Block::AIR);
}

int posToIndex(const int x, const int y, const int z)
{
    return x + 16 * z + 256 * y;
}

int posToIndex(const ivec3 pos)
{
    return posToIndex(pos.x, pos.y, pos.z);
}

void Chunk::dummyFill()
{
    for (int z = 0; z < 16; ++z)
    {
        for (int x = 0; x < 16; ++x)
        {
            int height = 48 + (rand() / (float)RAND_MAX * 16);
            for (int y = 0; y < height; ++y)
            {
                this->blocks[posToIndex(x, y, z)] = Block::STONE;
            }
        }
    }
}

static const std::array<ivec3, 6> directions = {
    ivec3(0, 0, 1), // forward
    ivec3(0, 0, -1), // back
    ivec3(1, 0, 0), // right
    ivec3(-1, 0, 0), // left
    ivec3(0, 1, 0), // up
    ivec3(0, -1, 0) // down
};

static const std::array<vec3, 24> directionVertPositions = {
    vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1),
    vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0),
    vec3(1, 0, 1), vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1),
    vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0),
    vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0), vec3(0, 1, 0),
    vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1)
};

static const std::array<vec2, 4> uvOffsets = {
    vec2(0, 0), vec2(0.0625f, 0), vec2(0.0625f, 0.0625f), vec2(0, 0.0625f)
};

void Chunk::createVBOs()
{
    idx.clear();
    verts.clear();

    idxCount = 0;

    for (int y = 0; y < 256; ++y)
    {
        for (int z = 0; z < 16; ++z)
        {
            for (int x = 0; x < 16; ++x)
            {
                ivec3 thisPos = ivec3(x, y, z);
                Block thisBlock = blocks[posToIndex(thisPos)];

                if (thisBlock == Block::AIR)
                {
                    continue;
                }

                BlockData thisBlockData = BlockUtils::getBlockData(thisBlock);

                for (int i = 0; i < 6; ++i)
                {
                    const auto& direction = directions[i];
                    const ivec3 newPos = thisPos + direction;
                    if (newPos.x < 0 || newPos.x >= 16 || newPos.z < 0 || newPos.z >= 16 || newPos.y < 0 || newPos.y >= 256)
                    {
                        continue;
                    }

                    Block neighborBlock = blocks[posToIndex(newPos)];
                    if (neighborBlock != Block::AIR)
                    {
                        continue;
                    }

                    int idx1 = verts.size();

                    vec2 uvStart;
                    switch (direction.y)
                    {
                    case 1:
                        uvStart = thisBlockData.uvs.top;
                        break;
                    case -1:
                        uvStart = thisBlockData.uvs.bottom;
                        break;
                    case 0:
                        uvStart = thisBlockData.uvs.side;
                        break;
                    }

                    for (int j = 0; j < 4; ++j)
                    {
                        verts.emplace_back();
                        Vertex& vert = verts.back();
                        vert.pos = vec3(thisPos) + directionVertPositions[i * 4 + j];
                        vert.uv = uvStart + uvOffsets[j];
                    }

                    idx.push_back(idx1);    
                    idx.push_back(idx1 + 1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1 + 3);
                }
            }
        }
    }
}

void Chunk::bufferVBOs()
{
    idxCount = idx.size();

    generateIdx();
    bindIdx();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    generateVerts();
    bindVerts();
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

    idx.clear();
    verts.clear();
}