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
    HAS_HEIGHTFIELD,
    NEEDS_LAYERS,
    HAS_LAYERS,
    NEEDS_EROSION,
    NEEDS_CAVES,
    NEEDS_FEATURE_PLACEMENTS,
    NEEDS_GATHER_FEATURE_PLACEMENTS,
    READY_TO_FILL, // this and 5x5 neighborhood all have feature placements
    FILLED,
    NEEDS_VBOS,
    DRAWABLE
};

class Chunk : public Drawable {
    template<std::size_t diameter>
    using ChunkProcessorFunc = std::function<void(Chunk* chunkPtr, Chunk* const (&neighborChunks)[diameter][diameter], int centerX, int centerZ)>;

private:
    ChunkState state{ ChunkState::EMPTY };
    bool readyForQueue{ true };

    std::vector<FeaturePlacement> featurePlacements;
    std::vector<FeaturePlacement> gatheredFeaturePlacements;
    std::vector<CaveFeaturePlacement> caveFeaturePlacements;
    std::vector<CaveFeaturePlacement> gatheredCaveFeaturePlacements;

public:
    const ivec2 worldChunkPos; // world space pos in terms of chunks (e.g. (3, -4) chunk pos = (48, -64) block pos)
    const ivec3 worldBlockPos;

    Zone* zonePtr;

    std::array<Chunk*, 4> neighbors{ nullptr };

    // TODO: use vector or something for heightfield and biomeWeights so they can be cleared after use (to save memory)
    // Will need to consider which things can't be cleared so simply (e.g. feature placements of thise chunk are used by other chunks too)
    // ===============================================================================
    // iteration order = z, x
    std::array<float, 256> heightfield;
    std::vector<float> gatheredHeightfield;
    
    // iteration order = y, z, x
    std::array<float, 256 * numMaterials> layers;

    // iteration order = z, x, y
    std::array<CaveLayer, 256 * MAX_CAVE_LAYERS_PER_COLUMN> caveLayers;

    // iteration order = y, z, x
    std::array<float, 256 * numBiomes> biomeWeights;

    // iteration order = z, x, y
    std::array<Block, 98304> blocks;
    // ===============================================================================

    std::vector<GLuint> idx;
    std::vector<Vertex> verts;

    Chunk(ivec2 worldChunkPos);

    ChunkState getState() const;
    void setState(ChunkState newState);
    bool isReadyForQueue();
    void setNotReadyForQueue();

private:
    template<std::size_t diameter>
    void floodFill(Chunk* (&neighborChunks)[diameter][diameter], ChunkState minState);
    template<std::size_t diameter>
    static void iterateNeighborChunks(Chunk* const (&neighborChunks)[diameter][diameter], ChunkState currentState, ChunkState nextState, ChunkProcessorFunc<diameter> chunkProcessorFunc);
    template<std::size_t diameter>
    void floodFillAndIterateNeighbors(ChunkState currentState, ChunkState nextState, ChunkProcessorFunc<diameter> chunkProcessorFunc);

    static void otherChunkGatherHeightfield(Chunk* chunkPtr, Chunk* const (&neighborChunks)[5][5], int centerX, int centerZ);

    void fixBackwardStratifiedLayers();

    static void otherChunkGatherFeaturePlacements(Chunk* chunkPtr, Chunk* const (&neighborChunks)[9][9], int centerX, int centerZ);

public:
    static void generateHeightfields(
        std::vector<Chunk*>& chunks,
        ivec2* host_chunkWorldBlockPositions,
        ivec2* dev_chunkWorldBlockPositions,
        float* host_heightfields,
        float* dev_heightfields,
        float* host_biomeWeights,
        float* dev_biomeWeights,
        cudaStream_t stream);

    void gatherHeightfield();

    static void generateLayers(
        std::vector<Chunk*>& chunks,
        float* host_heightfields,
        float* dev_heightfields,
        float* host_biomeWeights,
        float* dev_biomeWeights,
        ivec2* host_chunkWorldBlockPositions,
        ivec2* dev_chunkWorldBlockPositions,
        float* host_layers,
        float* dev_layers,
        cudaStream_t stream);

    static void erodeZone(
        Zone* zonePtr,
        float* host_gatheredLayers,
        float* dev_gatheredLayers, 
        float* dev_accumulatedHeights, 
        cudaStream_t stream);

    static void generateCaves(
        std::vector<Chunk*>& chunks,
        float* host_heightfields,
        float* dev_heightfields,
        float* host_biomeWeights,
        float* dev_biomeWeights,
        ivec2* host_chunkWorldBlockPositions,
        ivec2* dev_chunkWorldBlockPositions,
        CaveLayer* host_caveLayers,
        CaveLayer* dev_caveLayers,
        cudaStream_t stream);

    bool tryGenerateCaveFeaturePlacement(
        const CaveFeatureGen& caveFeatureGen,
        const CaveLayer& caveLayer,
        bool top,
        int caveFeaturePlacementSeed,
        float rand,
        ivec2 worldBlockPos2d);
    void generateColumnFeaturePlacements(int localX, int localZ);
    void generateFeaturePlacements();
    void gatherFeaturePlacements();

    static void fill(
        std::vector<Chunk*>& chunks,
        float* host_heightfields,
        float* dev_heightfields,
        float* host_biomeWeights,
        float* dev_biomeWeights,
        float* host_layers,
        float* dev_layers,
        CaveLayer* host_caveLayers,
        CaveLayer* dev_caveLayers,
        FeaturePlacement* dev_featurePlacements,
        CaveFeaturePlacement* dev_caveFeaturePlacements,
        Block* host_blocks,
        Block* dev_blocks,
        cudaStream_t stream);
    void tryPlaceSingleDecorator(
        ivec3 pos,
        const DecoratorGen& gen);
    void placeDecorators();

    void createVBOs();
    void bufferVBOs() override;
};
