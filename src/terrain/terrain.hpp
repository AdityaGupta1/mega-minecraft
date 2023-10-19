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

using namespace glm;

class Chunk;

struct Zone
{
    Zone(ivec2 worldChunkPos)
        : worldChunkPos(worldChunkPos)
    {}

    ivec2 worldChunkPos; // in terms of number of chunks, so (0, 0) then (0, 16) then (0, 32) and so on
    std::array<std::unique_ptr<Chunk>, 256> chunks{ nullptr };
    std::array<Zone*, 8> neighbors{ nullptr }; // starts with north and goes clockwise
};

class Terrain {
private:
    std::mutex mutex;

    std::unordered_map<ivec2, std::unique_ptr<Zone>, Utils::PosHash> zones;

    std::queue<Chunk*> chunksToGenerateHeightfield;
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

    void updateChunks();

    void createChunkVbos(Chunk* chunkPtr);

public:
    Terrain();
    ~Terrain();

    void tick();

    void draw(const ShaderProgram& prog);

    void setCurrentChunkPos(ivec2 newCurrentChunkPos);
};
