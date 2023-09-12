#include "terrain.hpp"

#define CHUNK_GEN_RADIUS 3

Terrain::Terrain()
{

}

void Terrain::updateChunks(ivec2 newCurrentChunkPos) 
{
    for (int dx = -CHUNK_GEN_RADIUS; dx <= CHUNK_GEN_RADIUS; ++dx)
    {
        for (int dz = -CHUNK_GEN_RADIUS; dz <= CHUNK_GEN_RADIUS; ++dz)
        {
            {
                ivec2 chunkPosToGenerate = newCurrentChunkPos + ivec2(dx, dz);
                ivec2 dist = chunkPosToGenerate - this->currentChunkPos;
                if (abs(dist.x) > CHUNK_GEN_RADIUS || abs(dist.y) > CHUNK_GEN_RADIUS && chunks.find(chunkPosToGenerate) == chunks.end())
                {
                    auto chunkPtr = std::make_unique<Chunk>();
                    auto chunkRawPtr = chunkPtr.get();
                    chunks[chunkPosToGenerate] = std::move(chunkPtr);
                    chunksToGenerate.push(chunkRawPtr);
                }
            }

            {
                ivec2 chunkPosToDestroy = this->currentChunkPos + ivec2(dx, dz);
                ivec2 dist = chunkPosToDestroy - newCurrentChunkPos;
                if (abs(dist.x) > CHUNK_GEN_RADIUS || abs(dist.y) > CHUNK_GEN_RADIUS && chunks.find(chunkPosToDestroy) != chunks.end())
                {
                    auto& chunkPtr = chunks[chunkPosToDestroy];
                    readyChunks.erase(chunkPosToDestroy);
                    // TODO destroy the chunk VBOs
                }
            }
        }
    }
}

void Terrain::tick() 
{
    // TODO do a kernel or something (based on queue of chunks to generate)
}

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunk) 
{
    if (this->currentChunkPos != newCurrentChunk) {
        updateChunks(newCurrentChunk);
        this->currentChunkPos = newCurrentChunk;
    }
}