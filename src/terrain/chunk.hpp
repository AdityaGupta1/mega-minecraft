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
    DRAWABLE
};

class Chunk : public Drawable {
private:
    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    void generateOwnFeaturePlacements();

public:
    const ivec2 worldChunkPos; // world-space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)
    const ivec3 worldBlockPos;

    Zone* zonePtr;

    std::array<Chunk*, 4> neighbors{ nullptr };

    // TODO: use vector or something for heightfield and biomeWeights so they can be cleared after use (to save memory)
    // Will need to consider which things can't be cleared so simply (e.g. feature placements of thise chunk are used by other chunks too)
    std::array<unsigned char, 256> heightfield; // iteration order = z, x
    std::array<float[(int)Biome::numBiomes], 256> biomeWeights; // iteration order = z, x
    std::array<Block, 65536> blocks; // iteration order = z, x, y (allows for easily copying horizontal slices of terrain)
    
    std::vector<FeaturePlacement> featurePlacements;

    std::vector<GLuint> idx;
    std::vector<Vertex> verts;

    Chunk(ivec2 worldChunkPos);

    ChunkState getState();
    void setState(ChunkState newState);
    bool isReadyForQueue();
    void setNotReadyForQueue();

    void generateHeightfield(unsigned char* dev_heightfield, float* dev_biomeWeights);
    void fill(Block* dev_blocks, unsigned char* dev_heightfield, float* dev_biomeWeights, FeaturePlacement* dev_featurePlacements);

    void createVBOs();
    void bufferVBOs() override;
};
