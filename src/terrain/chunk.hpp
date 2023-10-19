#pragma once

#include <glm/glm.hpp>
#include "block.hpp"
#include <array>
#include <vector>
#include "rendering/drawable.hpp"
#include "rendering/structs.hpp"
#include "terrain.hpp"
#include "biome.hpp"

using namespace glm;

struct Zone;

enum class ChunkState
{
    EMPTY,
    HAS_HEIGHTFIELD,
    HAS_FEATURE_PLACEMENTS, /* do on GPU if possible */
    IS_FILLED,
    HAS_VBOS,
    DRAWABLE
};

class Chunk : public Drawable {
private:
    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

public:
    const ivec2 worldChunkPos; // world-space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)

    Zone* zonePtr;

    std::array<Chunk*, 4> neighbors{ nullptr };

    // TODO: use vector or something for heightfield and biomeWeights so they can be cleared after use (to save memory)
    std::array<unsigned char, 256> heightfield; // iteration order = z, x
    std::array<float[(int)Biome::numBiomes], 256> biomeWeights;
    std::array<Block, 65536> blocks; // iteration order = z, x, y (allows for easily copying horizontal slices of terrain)

    std::vector<GLuint> idx;
    std::vector<Vertex> verts;

    Chunk(ivec2 worldChunkPos);

    ChunkState getState();
    void setState(ChunkState newState);
    bool isReadyForQueue();
    void setNotReadyForQueue();

    void dummyFillCUDA(Block* dev_blocks, unsigned char* dev_heightfield, float* dev_biomeWeights);

    void createVBOs();
    void bufferVBOs() override;
};
