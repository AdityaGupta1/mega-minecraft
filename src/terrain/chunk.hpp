#pragma once

#include <glm/glm.hpp>
#include "block.hpp"
#include <array>

using namespace glm;

enum class ChunkState
{
    EMPTY,
    HAS_HEIGHTFIELD,
    HAS_FEATURE_PLACEMENTS, /* do on GPU if possible */
    IS_FILLED,
    DRAWABLE
};

class Chunk {
public:
    const ivec2 worldChunkPos; // world-space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)

    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    std::array<unsigned char, 256> heightfield;
    std::array<Block, 65536> blocks;

    Chunk(ivec2 worldChunkPos);
};
