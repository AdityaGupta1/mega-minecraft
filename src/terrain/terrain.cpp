#include "terrain.hpp"

#include "util/enums.hpp"
#include "cuda/cuda_utils.hpp"
#include <thread>

#include <iostream>
#include <glm/gtx/string_cast.hpp>

#define CHUNK_VBOS_GEN_RADIUS 12
#define CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS (CHUNK_VBOS_GEN_RADIUS + 2)

#define MULTITHREADING 0

// --------------------------------------------------
#define TOTAL_ACTION_TIME 20
// --------------------------------------------------
#define ACTION_TIME_GENERATE_HEIGHTFIELD 5
#define ACTION_TIME_FILL 2
#define ACTION_TIME_CREATE_VBOS 5
#define ACTION_TIME_BUFFER_VBOS 20
// --------------------------------------------------

Terrain::Terrain()
{
    initCuda();
}

Terrain::~Terrain()
{
    freeCuda();
}

static constexpr int numDevBlocks = TOTAL_ACTION_TIME / ACTION_TIME_FILL;
static constexpr int numDevHeightfields = TOTAL_ACTION_TIME / min(ACTION_TIME_GENERATE_HEIGHTFIELD, ACTION_TIME_FILL);

static std::array<Block*, numDevBlocks> dev_blocks;
static std::array<unsigned char*, numDevHeightfields> dev_heightfields;
static std::array<float*, numDevHeightfields> dev_biomeWeights;

void Terrain::initCuda()
{
    for (int i = 0; i < numDevBlocks; ++i)
    {
        cudaMalloc((void**)&dev_blocks[i], 65536 * sizeof(Block));
    }

    for (int i = 0; i < numDevHeightfields; ++i)
    {
        cudaMalloc((void**)&dev_heightfields[i], 256 * sizeof(unsigned char));
        cudaMalloc((void**)&dev_biomeWeights[i], 256 * (int)Biome::numBiomes * sizeof(float));
    }

    CudaUtils::checkCUDAError("cudaMalloc failed");
}

void Terrain::freeCuda()
{
    for (int i = 0; i < numDevBlocks; ++i)
    {
        cudaFree(dev_blocks[i]);
    }

    for (int i = 0; i < numDevHeightfields; ++i)
    {
        cudaFree(dev_heightfields[i]);
        cudaFree(dev_biomeWeights[i]);
    }

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
                chunksToGenerateHeightfield.push(chunkPtr);
                continue;
            case ChunkState::HAS_HEIGHTFIELD:
                chunkPtr->setNotReadyForQueue();
                chunksToFill.push(chunkPtr);
                continue;
            }

            // don't create VBOs if not in range of CHUNK_VBOS_GEN_RADIUS
            const ivec2 dist = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
            if (max(dist.x, dist.y) > CHUNK_VBOS_GEN_RADIUS)
            {
                continue;
            }
            
            switch (chunkPtr->getState())
            {
            case ChunkState::IS_FILLED:
                chunkPtr->setNotReadyForQueue();
                chunksToCreateVbos.push(chunkPtr);
                continue;
            }
        }
    }
}

void Terrain::createChunkVbos(Chunk* chunkPtr)
{
    chunkPtr->createVBOs();

#if MULTITHREADING
    mutex.lock();
#endif

    chunkPtr->setNotReadyForQueue();
    chunksToBufferVbos.push(chunkPtr);

#if MULTITHREADING
    mutex.unlock();
#endif
}

void Terrain::tick()
{
#if MULTITHREADING
    mutex.lock();
#endif

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

    // TODO: temporary (?) system to weight VBO creation more than chunk filling
    // Will probably want to get better estimates for how much time these things take, especially as
    // terrain generation becomes more complicated. This also means that all chunks get filled before
    // any get VBOs created, which should help reduce the frequency of VBO recreation.
    int actionTimeLeft = TOTAL_ACTION_TIME;

    int blocksIdx = 0;
    int heightfieldIdx = 0;

    while (!chunksToGenerateHeightfield.empty() && actionTimeLeft >= ACTION_TIME_GENERATE_HEIGHTFIELD)
    {
        needsUpdateChunks = true;

        auto& chunkPtr = pop(chunksToGenerateHeightfield);

        chunkPtr->generateHeightfield(dev_heightfields[heightfieldIdx], dev_biomeWeights[heightfieldIdx]);
        ++heightfieldIdx;

        chunkPtr->setState(ChunkState::HAS_HEIGHTFIELD);

        actionTimeLeft -= ACTION_TIME_GENERATE_HEIGHTFIELD;
    }

    while (!chunksToFill.empty() && actionTimeLeft >= ACTION_TIME_FILL)
    {
        needsUpdateChunks = true;

        auto& chunkPtr = pop(chunksToFill);

        chunkPtr->fill(dev_blocks[blocksIdx], dev_heightfields[heightfieldIdx], dev_biomeWeights[heightfieldIdx]);
        ++blocksIdx;
        ++heightfieldIdx;

        chunkPtr->setState(ChunkState::IS_FILLED);

        for (int i = 0; i < 4; ++i)
        {
            Chunk* neighborChunkPtr = chunkPtr->neighbors[i];
            if (neighborChunkPtr != nullptr && neighborChunkPtr->getState() == ChunkState::DRAWABLE)
            {
                chunksToDestroyVbos.push(neighborChunkPtr);
            }
        }

        actionTimeLeft -= ACTION_TIME_FILL;
    }

    if (blocksIdx > 0 || heightfieldIdx > 0)
    {
        cudaDeviceSynchronize();
    }

#if MULTITHREADING
    while (!chunksToCreateVbos.empty())
#else
    while (!chunksToCreateVbos.empty() && actionTimeLeft >= ACTION_TIME_CREATE_VBOS)
#endif
    {
        needsUpdateChunks = true;

        auto& chunkPtr = pop(chunksToCreateVbos);

#if MULTITHREADING
        std::thread thread(&Terrain::createChunkVbos, this, chunkPtr);
        thread.detach();
#else
        this->createChunkVbos(chunkPtr);

        actionTimeLeft -= ACTION_TIME_CREATE_VBOS;
#endif
    }

    while (!chunksToBufferVbos.empty() && actionTimeLeft >= ACTION_TIME_BUFFER_VBOS)
    {
        auto& chunkPtr = pop(chunksToBufferVbos);

        chunkPtr->bufferVBOs();
        drawableChunks.insert(chunkPtr);
        chunkPtr->setState(ChunkState::DRAWABLE);

        actionTimeLeft -= ACTION_TIME_BUFFER_VBOS;
    }

#if MULTITHREADING
    mutex.unlock();
#endif
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