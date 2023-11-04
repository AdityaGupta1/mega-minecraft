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

#define ZONE_SIZE 4

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
};

class Terrain {
private:
    std::mutex mutex;

    std::unordered_map<ivec2, std::unique_ptr<Zone>, Utils::PosHash> zones;

    std::queue<Chunk*> chunksToGenerateHeightfield;
    std::queue<Chunk*> chunksToGatherHeightfield;
    std::queue<Chunk*> chunksToGenerateLayers;
    std::queue<Chunk*> chunksToGatherFeaturePlacements;
    std::queue<Chunk*> chunksToFill;
    std::queue<Chunk*> chunksToCreateVbos;
    std::queue<Chunk*> chunksToBufferVbos;
    
    std::unordered_set<Chunk*> drawableChunks;
    std::queue<Chunk*> chunksToDestroyVbos;

    ivec2 currentChunkPos{ 0, 0 };
    ivec2 lastChunkPos{ 0, 0 };
    bool needsUpdateChunks{ true };

    void initCuda();
    void freeCuda();

    Zone* createZone(ivec2 zonePos);

    void updateChunk(int dx, int dz);
    void updateChunks();

public:
    Terrain();
    ~Terrain();

    void tick();

    void draw(const ShaderProgram& prog, const Player* player);

    void setCurrentChunkPos(ivec2 newCurrentChunkPos);
};
