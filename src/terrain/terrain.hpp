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

#define ZONE_SIZE 12 // changing this may have disastrous consequences
#define EROSION_GRID_SIDE_LENGTH_BLOCKS (ZONE_SIZE * 2 * 16)
#define BLOCKS_PER_EROSION_KERNEL (EROSION_GRID_SIDE_LENGTH_BLOCKS * EROSION_GRID_SIDE_LENGTH_BLOCKS)

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

class Terrain {
private:
    std::mutex mutex;

    std::unordered_map<ivec2, std::unique_ptr<Zone>, Utils::PosHash> zones;

    std::queue<Chunk*> chunksToGenerateHeightfield;
    std::queue<Chunk*> chunksToGatherHeightfield;
    std::queue<Chunk*> chunksToGenerateLayers;
    std::queue<Chunk*> chunksToGatherFeaturePlacements;
    std::unordered_set<Zone*> zonesToTryErosion;
    std::queue<Zone*> zonesToErode;
    std::queue<Chunk*> chunksToFill;
    std::queue<Chunk*> chunksToCreateAndBufferVbos;
    
    std::unordered_set<Chunk*> drawableChunks;
    std::queue<Chunk*> chunksToDestroyVbos;

    ivec2 currentChunkPos{ 0, 0 };
    ivec2 lastChunkPos{ 0, 0 };
    bool needsUpdateChunks{ true };

    void initCuda();
    void freeCuda();

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

    Chunk* debugGetCurrentChunk();
    void debugPrintCurrentChunkState();
    void debugPrintCurrentColumnLayers(vec2 playerPos);
};
