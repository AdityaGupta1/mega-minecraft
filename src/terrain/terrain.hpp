#pragma once

#include "chunk.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <memory>
#include <array>
#include "util/utils.hpp"

using namespace glm;

struct Zone
{
    Zone(ivec2 worldChunkPos)
        : worldChunkPos(worldChunkPos), chunks(), neighbors()
    {
    }

    ivec2 worldChunkPos; // in terms of number of chunks, so (0, 0) then (0, 16) then (0, 32) and so on
    std::array<std::unique_ptr<Chunk>, 256> chunks;
    std::array<Zone*, 8> neighbors; // starts with north and goes clockwise
};

class Terrain {
private:
    std::unordered_map<ivec2, std::unique_ptr<Zone>, Utils::PosHash> zones;

    //std::queue<Chunk*> chunksToGenerate;
    //std::queue<Chunk*> chunksThatNeedVbos;
    
    //std::unordered_set<Chunk*> drawableChunks;

    ivec2 currentChunkPos;

    Zone* createZone(ivec2 zonePos);

    void updateChunks();

public:
    Terrain();

    void tick();

    void draw();

    void setCurrentChunkPos(ivec2 newCurrentChunkPos);
};
