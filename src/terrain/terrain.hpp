#pragma once

#include "chunk.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <queue>
#include <memory>
#include "util/utils.hpp"

using namespace glm;

class Terrain {
private:
    std::unordered_map<ivec2, std::unique_ptr<Chunk>, Utils::PosHash> chunks;

    std::queue<Chunk*> chunksToGenerate;
    std::queue<Chunk*> chunksThatNeedVbos;
    
    std::unordered_map<ivec2, std::unique_ptr<Chunk>, Utils::PosHash> readyChunks;

    ivec2 currentChunkPos;

    void updateChunks(ivec2 newCurrentChunkPos);

public:
    Terrain();

    void tick();

    void setCurrentChunkPos(ivec2 newCurrentChunkPos);
};