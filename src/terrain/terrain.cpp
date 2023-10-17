#include "terrain.hpp"

#include "util/enums.hpp"
#include "cuda/cuda_utils.hpp"

#include <iostream>
#include <glm/gtx/string_cast.hpp>

#define CHUNK_VBOS_GEN_RADIUS 8
#define CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS (CHUNK_VBOS_GEN_RADIUS + 2)

Terrain::Terrain()
{
    initCuda();
}

Terrain::~Terrain()
{
    freeCuda();
}

static Block* dev_blocks;
static unsigned char* dev_heightfield;

void Terrain::initCuda()
{
    cudaMalloc((void**)&dev_blocks, 65536 * sizeof(Block));
    cudaMalloc((void**)&dev_heightfield, 256 * sizeof(unsigned char));
    CudaUtils::checkCUDAError("cudaMalloc failed");
}

void Terrain::freeCuda()
{
    cudaFree(dev_blocks);
    cudaFree(dev_heightfield);
    CudaUtils::checkCUDAError("cudaFree failed");
}

ivec2 zonePosFromChunkPos(ivec2 chunkPos)
{
    return ivec2(glm::floor(vec2(chunkPos) / 16.f)) * 16;
}

int localChunkPosToIdx(ivec2 localChunkPos)
{
    return localChunkPos.x + 16 * localChunkPos.y;
}

template<class T>
T& pop(std::queue<T>& queue)
{
    T& temp = queue.front();
    queue.pop();
    return temp;
}

Zone* Terrain::createZone(ivec2 zonePos)
{
    auto newZoneUptr = std::make_unique<Zone>(zonePos);
    Zone* newZonePtr = newZoneUptr.get();
    zones[zonePos] = std::move(newZoneUptr);

    for (int i = 0; i < 8; ++i)
    {
        ivec2 neighborPos = zonePos + (16 * DirectionEnums::dirVecs2D[i]);

        auto neighborZoneIt = zones.find(neighborPos);
        if (neighborZoneIt == zones.end())
        {
            continue;
        }

        auto neighborZonePtr = neighborZoneIt->second.get();

        newZonePtr->neighbors[i] = neighborZonePtr;
        neighborZonePtr->neighbors[(i + 4) % 8] = newZonePtr;
    }

    return newZonePtr;
}

// This function looks at the area covered by CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS. For each chunk, it first
// creates the chunk if it doesn't exist, also creating the associated zone if necsesary. Then, it looks at
// the chunk's state and adds it to the correct queue iff chunk.readyForQueue == true. This also includes
// setting readyForQueue to false so the chunk will be skipped until it's ready for the next state. The states 
// themselves, as well as readyForQueue, will be updated after the associated call or CPU thread finishes 
// execution. Kernels and threads are spawned from Terrain::tick().
void Terrain::updateChunks()
{
    for (int dz = -CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; dz <= CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; ++dz)
    {
        for (int dx = -CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; dx <= CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; ++dx)
        {
            ivec2 newChunkWorldChunkPos = currentChunkPos + ivec2(dx, dz);
            ivec2 newZoneWorldChunkPos = zonePosFromChunkPos(newChunkWorldChunkPos);

            auto zoneIt = zones.find(newZoneWorldChunkPos);
            Zone* zonePtr; // TODO maybe cache this and reuse if next chunk has same zone (which should be a common occurrence)
            if (zoneIt == zones.end())
            {
                zonePtr = createZone(newZoneWorldChunkPos);
            }
            else
            {
                zonePtr = zoneIt->second.get();
            }

            ivec2 newChunkLocalChunkPos = newChunkWorldChunkPos - newZoneWorldChunkPos;
            int chunkIdx = localChunkPosToIdx(newChunkLocalChunkPos);

            if (zonePtr->chunks[chunkIdx] == nullptr)
            {
                auto chunkUptr = std::make_unique<Chunk>(newChunkWorldChunkPos);
                chunkUptr->zonePtr = zonePtr;

                for (int i = 0; i < 4; ++i)
                {
                    const auto& neighborChunkDir = DirectionEnums::dirVecs[i];
                    const ivec2 neighborChunkLocalChunkPos = newChunkLocalChunkPos 
                        + ivec2(neighborChunkDir.x, neighborChunkDir.z);

                    Zone* neighborZonePtr = zonePtr;
                    if (neighborChunkLocalChunkPos.x < 0 || neighborChunkLocalChunkPos.x >= 16
                        || neighborChunkLocalChunkPos.y < 0 || neighborChunkLocalChunkPos.y >= 16)
                    {
                        neighborZonePtr = zonePtr->neighbors[i * 2];

                        if (neighborZonePtr == nullptr)
                        {
                            continue;
                        }
                    }

                    const int neighborChunkIdx = localChunkPosToIdx((neighborChunkLocalChunkPos + ivec2(16)) % 16);
                    const auto& neighborChunkUptr = neighborZonePtr->chunks[neighborChunkIdx];
                    if (neighborChunkUptr == nullptr)
                    {
                        continue;
                    }

                    chunkUptr->neighbors[i] = neighborChunkUptr.get();
                    neighborChunkUptr->neighbors[(i + 2) % 4] = chunkUptr.get();
                }

                zonePtr->chunks[chunkIdx] = std::move(chunkUptr);
            }

            Chunk* chunkPtr = zonePtr->chunks[chunkIdx].get();

            if (!chunkPtr->isReadyForQueue())
            {
                continue;
            }

            switch (chunkPtr->getState())
            {
            case ChunkState::EMPTY:
                chunkPtr->setNotReadyForQueue();
                dummyChunksToFill.push(chunkPtr);
                continue;
            }

            // don't go past feature placements if not in range of CHUNK_VBOS_GEN_RADIUS
            const ivec2 dist = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
            if (max(dist.x, dist.y) > CHUNK_VBOS_GEN_RADIUS)
            {
                continue;
            }
            
            switch (chunkPtr->getState())
            {
            case ChunkState::IS_FILLED:
                chunkPtr->setNotReadyForQueue();
                dummyChunksToCreateVbos.push(chunkPtr);
                continue;
            }
        }
    }
}

void Terrain::tick()
{
    while (!chunksToDestroyVbos.empty())
    {
        auto& chunkPtr = pop(chunksToDestroyVbos);

        drawableChunks.erase(chunkPtr);
        chunkPtr->destroyVBOs();
        chunkPtr->setState(ChunkState::IS_FILLED);
    }

    if (currentChunkPos != lastChunkPos)
    {
        lastChunkPos = currentChunkPos;
        needsUpdateChunks = true;
    }

    if (needsUpdateChunks)
    {
        updateChunks();
        needsUpdateChunks = false;
    }

    while (!dummyChunksToFill.empty())
    {
        needsUpdateChunks = true;

        auto& chunkPtr = pop(dummyChunksToFill);

        chunkPtr->dummyFillCUDA(dev_blocks, dev_heightfield);
        chunkPtr->setState(ChunkState::IS_FILLED);

        for (int i = 0; i < 4; ++i)
        {
            Chunk* neighborChunkPtr = chunkPtr->neighbors[i];
            if (neighborChunkPtr != nullptr && neighborChunkPtr->getState() == ChunkState::DRAWABLE)
            {
                chunksToDestroyVbos.push(neighborChunkPtr);
            }
        }

        break; // temporary so it does only one per frame
    }

    while (!dummyChunksToCreateVbos.empty())
    {
        auto& chunkPtr = pop(dummyChunksToCreateVbos);

        chunkPtr->createVBOs();
        chunkPtr->bufferVBOs();
        drawableChunks.insert(chunkPtr);
        chunkPtr->setState(ChunkState::DRAWABLE);

        break; // temporary so it does only one per frame
    }

    // TODO do a kernel or launch a thread or something (based on queue of chunks to generate)
    // go through all the queues and do kernels/threads (up to a max number of kernels, probably no limit on queueing threads)
    // max number of kernels may depend on type and measured execution type of each kernel
    // make sure to update chunk.readyForQueue to true afterwards
    // IMPORTANT: VBO threads should, when finished, directly add chunks to some set of "chunks that need VBOs buffered" to prevent leaking chunks that become too far
}

void Terrain::draw(const ShaderProgram& prog) {
    mat4 modelMat = mat4(1);

    for (const auto& chunkPtr : drawableChunks)
    {
        const ivec2 dist = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
        if (max(dist.x, dist.y) > CHUNK_VBOS_GEN_RADIUS)
        {
            chunksToDestroyVbos.push(chunkPtr);
        }
        else
        {
            modelMat[3][0] = chunkPtr->worldChunkPos.x * 16.f;
            modelMat[3][2] = chunkPtr->worldChunkPos.y * 16.f;
            prog.setModelMat(modelMat);
            prog.draw(*chunkPtr);
        }
    }
}

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunk) 
{
    this->currentChunkPos = newCurrentChunk;
}