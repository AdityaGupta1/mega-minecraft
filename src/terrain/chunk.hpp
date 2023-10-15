#pragma once

#include <glm/glm.hpp>
#include "block.hpp"
#include <array>
#include <vector>
#include "rendering/drawable.hpp"
#include "rendering/structs.hpp"
#include "terrain.hpp"

using namespace glm;

struct Zone;

enum class ChunkState
{
    EMPTY,
    HAS_HEIGHTFIELD,
    HAS_FEATURE_PLACEMENTS, /* do on GPU if possible */
    IS_FILLED,
    DRAWABLE
};

class Chunk : public Drawable {
public:
    const ivec2 worldChunkPos; // world-space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)

    Zone* zonePtr;

    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    std::array<unsigned char, 256> heightfield;
    std::array<Block, 65536> blocks; // iteration order = z, x, y (allows for easily copying horizontal slices of terrain)

    std::vector<GLuint> idx;
    std::vector<Vertex> verts;

    Chunk(ivec2 worldChunkPos);

    void dummyFill();

    void createVBOs();
    void bufferVBOs() override;
};
