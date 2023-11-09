#pragma once

#include "chunk.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <memory>
#include <array>
#include "util/utils.hpp"
#include "rendering/shaderProgram.hpp"
#include <mutex>
#include "player/player.hpp"
#include "biome.hpp"

#define ZONE_SIZE 12 // changing this may have disastrous consequences
#define EROSION_GRID_SIDE_LENGTH_BLOCKS (ZONE_SIZE * 2 * 16)
#define COLS_PER_EROSION_KERNEL (EROSION_GRID_SIDE_LENGTH_BLOCKS * EROSION_GRID_SIDE_LENGTH_BLOCKS)

using namespace glm;

class Chunk;

struct Zone
{
    Zone(ivec2 worldChunkPos)
        : worldChunkPos(worldChunkPos)
    {}

    ivec2 worldChunkPos; // in terms of number of chunks, so (0, 0) then (0, 1 * ZONE_SIZE) then (0, 2 * ZONE_SIZE) and so on
    std::array<std::unique_ptr<Chunk>, ZONE_SIZE * ZONE_SIZE> chunks{ nullptr };
    std::array<Zone*, 8> neighbors{ nullptr }; // starts with north and goes clockwise

    std::vector<Chunk*> gatheredChunks;
    bool hasBeenQueuedForErosion{ false };
};

static constexpr int devBlocksSize = 16 * 384 * 16;
static constexpr int devFeaturePlacementsSize = MAX_GATHERED_FEATURES_PER_CHUNK;

static constexpr int devHeightfieldSize = 18 * 18;
static constexpr int devBiomeWeightsSize = 256 * numBiomes;

static constexpr int devLayersSize = 256 * numMaterials;

static constexpr int devGatheredLayersSize = COLS_PER_EROSION_KERNEL * (numErodedMaterials + 1) + 1;
static constexpr int devAccumulatedHeightsSize = COLS_PER_EROSION_KERNEL;

class Terrain {
private:
    std::vector<ivec2> spiral;

    std::unordered_map<ivec2, std::unique_ptr<Zone>, Utils::PosHash> zones;

    std::queue<Chunk*> chunksToGenerateHeightfield;
    std::queue<Chunk*> chunksToGatherHeightfield;
    std::queue<Chunk*> chunksToGenerateLayers;
    std::unordered_set<Zone*> zonesToTryErosion;
    std::queue<Zone*> zonesToErode;
    std::queue<Chunk*> chunksToGenerateFeaturePlacements;
    std::queue<Chunk*> chunksToGatherFeaturePlacements;
    std::queue<Chunk*> chunksToFill;
    std::queue<Chunk*> chunksToCreateAndBufferVbos;
    
    std::unordered_set<Chunk*> drawableChunks;
    std::queue<Chunk*> chunksToDestroyVbos;

    ivec2 currentChunkPos{ 0, 0 };
    ivec2 lastChunkPos{ 0, 0 };
    bool needsUpdateChunks{ true };

    Zone* lastUpdateZonePtr{ nullptr };

    void initCuda();
    void freeCuda();

    void generateSpiral();

    Zone* createZone(ivec2 zoneWorldChunkPos);

    void updateChunk(int dx, int dz);
    void updateChunks();

    void addZonesToTryErosionSet(Chunk* chunkPtr);
    void updateZones();

public:
    Terrain();
    ~Terrain();

    void tick();

    void draw(const ShaderProgram& prog, const Player* player);

    ivec2 getCurrentChunkPos() const;
    void setCurrentChunkPos(ivec2 newCurrentChunkPos);

    void debugGetCurrentChunkAndZone(vec2 playerPos, Chunk** chunkPtr, Zone** zonePtr);
    void debugPrintCurrentChunkState(vec2 playerPos);
    void debugPrintCurrentColumnLayers(vec2 playerPos);
};
