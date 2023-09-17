#pragma once

#include <glm/glm.hpp>

using namespace glm;

enum class ChunkState
{
    EMPTY,
    HAS_HEIGHTFIELD,
    HAS_FEATURE_PLACEMENTS,
    HAS_FEATURES,
    DRAWABLE
};

class Chunk {
public:
    const ivec2 worldChunkPos; // world-space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)

    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    Chunk(ivec2 worldChunkPos);
};
