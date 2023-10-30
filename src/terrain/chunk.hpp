#pragma once

#include <glm/glm.hpp>
#include "block.hpp"
#include <array>
#include <vector>
#include "rendering/drawable.hpp"
#include "rendering/structs.hpp"
#include "terrain.hpp"
#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <functional>

using namespace glm;

struct Zone;

enum class ChunkState : unsigned char
{
    EMPTY,
    HAS_HEIGHTFIELD_AND_FEATURE_PLACEMENTS,
    READY_TO_FILL, // this and 5x5 neighborhood all have feature placements
    IS_FILLED,
    DRAWABLE
};

class Chunk : public Drawable {
    template<std::size_t diameter>
    using ChunkProcessorFunc = std::function<bool(Chunk* chunkPtr, Chunk* const (&neighborChunks)[diameter][diameter], int centerX, int centerZ)>;

private:
    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    std::vector<FeaturePlacement> featurePlacements;
    std::vector<FeaturePlacement> gatheredFeaturePlacements;

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

    std::vector<GLuint> idx;
    std::vector<Vertex> verts;

    Chunk(ivec2 worldChunkPos);

    ChunkState getState();
    void setState(ChunkState newState);
    bool isReadyForQueue();
    void setNotReadyForQueue();

    void generateHeightfield(unsigned char* dev_heightfield, float* dev_biomeWeights, cudaStream_t stream);

private:
    template<std::size_t diameter>
    void floodFill(Chunk* (&neighborChunks)[diameter][diameter], ChunkState minState);
    template<std::size_t diameter>
    static void iterateNeighborChunks(Chunk* const (&neighborChunks)[diameter][diameter], ChunkState currentState, ChunkState nextState, ChunkProcessorFunc<diameter> chunkProcessorFunc);
    template<std::size_t diameter>
    void floodFillAndIterateNeighbors(ChunkState currentState, ChunkState nextState, ChunkProcessorFunc<diameter> chunkProcessorFunc);

    static bool otherChunkGatherFeaturePlacements(Chunk* chunkPtr, Chunk* const (&neighborChunks)[9][9], int centerX, int centerZ);

public:
    void gatherFeaturePlacements();

    void fill(Block* dev_blocks, unsigned char* dev_heightfield, float* dev_biomeWeights, FeaturePlacement* dev_featurePlacements, cudaStream_t stream);

    void createVBOs();
    void bufferVBOs() override;
};
