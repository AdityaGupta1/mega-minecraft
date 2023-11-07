#include "terrain.hpp"

#include "util/enums.hpp"
#include "cuda/cudaUtils.hpp"
#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <glm/gtx/string_cast.hpp>

#define DEBUG_TIME_CHUNK_FILL 0

// ================================================================================================================================================================
// theoretical padding needed:
// [+1] gather heightfields of 3x3 chunks and place material layers
// [+5] gather chunks to do zone erosion (for a corner chunk, 3 more in zone and 2 for padding; may not be super accurate but whatever)
//     [+2] gather feature placements of 5x5 chunks for filling chunk (this is independent of erosion so it can be contained in the +5 for erosion)
// ================================================================================================================================================================
// in practice, I give it way more padding so the user doesn't see any lag at borders of render distance
// ================================================================================================================================================================
static constexpr int chunkVbosGenRadius = 20;
static constexpr int chunkMaxGenRadius = chunkVbosGenRadius + ((ZONE_SIZE * 5) / 2);

// TODO: get better estimates for these
// ================================================================================
static constexpr int totalActionTime = 500;
// ================================================================================
static constexpr int actionTimeGenerateHeightfield        = 4;
static constexpr int actionTimeGatherHeightfield          = 2;
static constexpr int actionTimeGenerateLayers             = 6;
static constexpr int actionTimeErodeZone                  = totalActionTime;
static constexpr int actionTimeGatherFeaturePlacements    = 2;
static constexpr int actionTimeFill                       = 4;
static constexpr int actionTimeCreateAndBufferVbos        = totalActionTime / 4;
// ================================================================================

Terrain::Terrain()
{
    initCuda();
}

Terrain::~Terrain()
{
    freeCuda();
}

#if DEBUG_TIME_CHUNK_FILL
bool startedTiming = false;
bool finishedTiming = false;
std::chrono::system_clock::time_point start;
#endif

static constexpr int numDevBlocks = totalActionTime / actionTimeFill;
static constexpr int numDevHeightfields = totalActionTime / min(actionTimeGenerateHeightfield, min(actionTimeGenerateLayers, actionTimeFill));
static constexpr int numDevLayers = totalActionTime / min(actionTimeGenerateLayers, actionTimeFill);
static constexpr int numDevGatheredLayers = totalActionTime / actionTimeErodeZone;
static constexpr int numStreams = max(max(numDevBlocks, numDevHeightfields), max(numDevLayers, numDevGatheredLayers));

static std::array<Block*, numDevBlocks> dev_blocks;
static std::array<FeaturePlacement*, numDevBlocks> dev_featurePlacements;

static std::array<float*, numDevHeightfields> dev_heightfields;
static std::array<float*, numDevHeightfields> dev_biomeWeights; // TODO: may need to set numDevBiomeWeights = max(numDevHeightfields, numDevLayers)
                                                                // probably fine for now since biome weights are always used in tandem with heightfield

static std::array<float*, numDevLayers> dev_layers;

static std::array<float*, numDevGatheredLayers> dev_gatheredLayers;
static std::array<float*, numDevGatheredLayers> dev_accumulatedHeights;

static std::array<cudaStream_t, numStreams> streams;

void Terrain::initCuda()
{
    for (int i = 0; i < numDevBlocks; ++i)
    {
        cudaMalloc((void**)&dev_blocks[i], 98304 * sizeof(Block));
        cudaMalloc((void**)&dev_featurePlacements[i], MAX_FEATURES_PER_CHUNK * sizeof(FeaturePlacement));
    }

    for (int i = 0; i < numDevHeightfields; ++i)
    {
        cudaMalloc((void**)&dev_heightfields[i], 18 * 18 * sizeof(float));
        cudaMalloc((void**)&dev_biomeWeights[i], 256 * numBiomes * sizeof(float));
    }

    for (int i = 0; i < numDevLayers; ++i)
    {
        cudaMalloc((void**)&dev_layers[i], 256 * numMaterials * sizeof(float));
    }

    for (int i = 0; i < numDevGatheredLayers; ++i)
    {
        cudaMalloc((void**)&dev_gatheredLayers[i], (BLOCKS_PER_EROSION_KERNEL * (numErodedMaterials + 1) + 1) * sizeof(float));
        cudaMalloc((void**)&dev_accumulatedHeights[i], BLOCKS_PER_EROSION_KERNEL * sizeof(float));
    }

    CudaUtils::checkCUDAError("cudaMalloc failed");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    CudaUtils::checkCUDAError("cudaStreamCreate failed");
}

void Terrain::freeCuda()
{
    for (int i = 0; i < numDevBlocks; ++i)
    {
        cudaFree(dev_blocks[i]);
        cudaFree(dev_featurePlacements[i]);
    }

    for (int i = 0; i < numDevHeightfields; ++i)
    {
        cudaFree(dev_heightfields[i]);
        cudaFree(dev_biomeWeights[i]);
    }

    for (int i = 0; i < numDevLayers; ++i)
    {
        cudaFree(dev_layers[i]);
    }

    for (int i = 0; i < numDevGatheredLayers; ++i)
    {
        cudaFree(dev_gatheredLayers[i]);
        cudaFree(dev_accumulatedHeights[i]);
    }

    CudaUtils::checkCUDAError("cudaFree failed");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    CudaUtils::checkCUDAError("cudaStreamDestroy failed");
}

ivec2 zonePosFromChunkPos(ivec2 chunkPos)
{
    return ivec2(glm::floor(vec2(chunkPos) / (float)ZONE_SIZE)) * ZONE_SIZE;
}

template<int size = ZONE_SIZE>
int localChunkPosToIdx(int x, int z)
{
    return x + size * z;
}

template<int size = ZONE_SIZE>
int localChunkPosToIdx(ivec2 localChunkPos)
{
    return localChunkPosToIdx<size>(localChunkPos.x, localChunkPos.y);
}

Zone* Terrain::createZone(ivec2 zoneWorldChunkPos)
{
    auto newZoneUptr = std::make_unique<Zone>(zoneWorldChunkPos);
    Zone* newZonePtr = newZoneUptr.get();
    zones[zoneWorldChunkPos] = std::move(newZoneUptr);

    for (int i = 0; i < 8; ++i)
    {
        ivec2 neighborPos = zoneWorldChunkPos + (ZONE_SIZE * DirectionEnums::dirVecs2d[i]);

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

void Terrain::updateChunk(int dx, int dz)
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
            if (neighborChunkLocalChunkPos.x < 0 || neighborChunkLocalChunkPos.x >= ZONE_SIZE
                || neighborChunkLocalChunkPos.y < 0 || neighborChunkLocalChunkPos.y >= ZONE_SIZE)
            {
                neighborZonePtr = zonePtr->neighbors[i * 2];

                if (neighborZonePtr == nullptr)
                {
                    continue;
                }
            }

            const int neighborChunkIdx = localChunkPosToIdx((neighborChunkLocalChunkPos + ivec2(ZONE_SIZE)) % ZONE_SIZE);
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
        return;
    }

    switch (chunkPtr->getState())
    {
    case ChunkState::EMPTY:
        chunkPtr->setNotReadyForQueue();
        chunksToGenerateHeightfield.push(chunkPtr);
        return;
    case ChunkState::HAS_HEIGHTFIELD:
        chunkPtr->setNotReadyForQueue();
        chunksToGatherHeightfield.push(chunkPtr);
        return;
    case ChunkState::NEEDS_LAYERS:
        chunkPtr->setNotReadyForQueue();
        chunksToGenerateLayers.push(chunkPtr);
        return;
    case ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS:
        chunkPtr->setNotReadyForQueue();
        chunksToGatherFeaturePlacements.push(chunkPtr);
        return;
    case ChunkState::READY_TO_FILL:
        chunkPtr->setNotReadyForQueue();
        chunksToFill.push(chunkPtr);
        return;
    }

    const ivec2 dist = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
    if (max(dist.x, dist.y) > chunkVbosGenRadius)
    {
        return;
    }

    switch (chunkPtr->getState())
    {
    case ChunkState::NEEDS_VBOS:
        chunkPtr->setNotReadyForQueue();
        chunksToCreateAndBufferVbos.push(chunkPtr);
        return;
    }
}

void Terrain::updateChunks()
{
    for (int dz = -chunkMaxGenRadius; dz <= chunkMaxGenRadius; ++dz)
    {
        for (int dx = -chunkMaxGenRadius; dx <= chunkMaxGenRadius; ++dx)
        {
            updateChunk(dx, dz);
        }
    }
}

void Terrain::addZonesToTryErosionSet(Chunk* chunkPtr)
{
    Zone* zonePtr = chunkPtr->zonePtr;
    zonesToTryErosion.insert(zonePtr);

    ivec2 localChunkPos = chunkPtr->worldChunkPos - zonePtr->worldChunkPos;
    int startDirIdx;
    if (localChunkPos.x < ZONE_SIZE / 2)
    {
        startDirIdx = localChunkPos.y < ZONE_SIZE / 2 ? 4 : 6;
    }
    else
    {
        startDirIdx = localChunkPos.y < ZONE_SIZE / 2 ? 0 : 2;
    }

    for (int i = 0; i < 3; ++i)
    {
        Zone* neighborZonePtr = zonePtr->neighbors[(startDirIdx + i) % 8];
        if (neighborZonePtr != nullptr && !neighborZonePtr->hasBeenQueuedForErosion)
        {
            zonesToTryErosion.insert(neighborZonePtr);
        }
    }
}

ivec2 getNeighborZoneCornerCoordBounds(int offset)
{
    switch (offset)
    {
    case -1:
        return ivec2(ZONE_SIZE / 2, ZONE_SIZE);
    case 0:
        return ivec2(0, ZONE_SIZE);
    case 1:
        return ivec2(0, ZONE_SIZE / 2);
    }

    throw std::exception("invalid offset");
}

bool isChunkReadyForErosion(Chunk* chunkPtr, Zone* zonePtr)
{
    if (chunkPtr == nullptr || chunkPtr->getState() < ChunkState::HAS_LAYERS)
    {
        return false;
    }

    ivec2 gatheredChunkPos = chunkPtr->worldChunkPos - zonePtr->worldChunkPos + ivec2(ZONE_SIZE / 2);
    zonePtr->gatheredChunks[localChunkPosToIdx<ZONE_SIZE * 2>(gatheredChunkPos)] = chunkPtr;
    return true;
}

bool isZoneReadyForErosion(Zone* zonePtr)
{
    for (const auto& chunkPtr : zonePtr->chunks)
    {
        if (!isChunkReadyForErosion(chunkPtr.get(), zonePtr))
        {
            return false;
        }
    }

    for (int i = 0; i < 8; ++i)
    {
        const Zone* neighborZonePtr = zonePtr->neighbors[i];

        if (neighborZonePtr == nullptr)
        {
            continue;
        }

        const ivec2 neighborDir = DirectionEnums::dirVecs2d[i];
        ivec2 xBounds = getNeighborZoneCornerCoordBounds(neighborDir.x);
        ivec2 zBounds = getNeighborZoneCornerCoordBounds(neighborDir.y);

        for (int z = zBounds[0]; z < zBounds[1]; ++z)
        {
            for (int x = xBounds[0]; x < xBounds[1]; ++x)
            {
                const auto& chunkPtr = neighborZonePtr->chunks[localChunkPosToIdx(x, z)];
                if (!isChunkReadyForErosion(chunkPtr.get(), zonePtr))
                {
                    return false;
                }
            }
        }
    }

    return true;
}

void Terrain::updateZones()
{
    for (const auto& zonePtr : zonesToTryErosion)
    {
        zonePtr->gatheredChunks.reserve(ZONE_SIZE * ZONE_SIZE * 4);
        if (isZoneReadyForErosion(zonePtr))
        {
            zonesToErode.push(zonePtr);
            zonePtr->hasBeenQueuedForErosion = true;
        }
        else
        {
            zonePtr->gatheredChunks.clear();
        }
    }

    zonesToTryErosion.clear();
}

void checkChunkAndNeighborsForNeedsVbos(Chunk* chunkPtr)
{
    if (chunkPtr == nullptr || chunkPtr->getState() < ChunkState::FILLED)
    {
        return;
    }

    for (const auto& neighborChunkPtr : chunkPtr->neighbors)
    {
        if (neighborChunkPtr == nullptr || neighborChunkPtr->getState() < ChunkState::FILLED)
        {
            return;
        }
    }

    chunkPtr->setState(ChunkState::NEEDS_VBOS);
}

void Terrain::tick()
{
    while (!chunksToDestroyVbos.empty())
    {
        auto chunkPtr = chunksToDestroyVbos.front();
        chunksToDestroyVbos.pop();

        drawableChunks.erase(chunkPtr);
        chunkPtr->destroyVBOs();
        chunkPtr->setState(ChunkState::NEEDS_VBOS);
    }

    if (currentChunkPos != lastChunkPos)
    {
        lastChunkPos = currentChunkPos;
        needsUpdateChunks = true;
    }

    if (needsUpdateChunks)
    {
        updateChunks();
        updateZones();
        needsUpdateChunks = false;
    }

    int actionTimeLeft = totalActionTime;

    int blocksIdx = 0;
    int heightfieldIdx = 0;
    int layersIdx = 0;
    int gatheredLayersIdx = 0;
    int streamIdx = 0;

    while (!chunksToCreateAndBufferVbos.empty() && actionTimeLeft >= actionTimeCreateAndBufferVbos)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToCreateAndBufferVbos.front();
        chunksToCreateAndBufferVbos.pop();

        chunkPtr->createVBOs();
        chunkPtr->bufferVBOs();
        drawableChunks.insert(chunkPtr);
        chunkPtr->setState(ChunkState::DRAWABLE);
        chunkPtr->setNotReadyForQueue();

        actionTimeLeft -= actionTimeCreateAndBufferVbos;
    }

    while (!chunksToFill.empty() && actionTimeLeft >= actionTimeFill)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToFill.front();
        chunksToFill.pop();

        chunkPtr->fill(
            dev_blocks[blocksIdx],
            dev_heightfields[heightfieldIdx],
            dev_layers[layersIdx],
            dev_biomeWeights[heightfieldIdx],
            dev_featurePlacements[blocksIdx],
            streams[streamIdx]
        );
        ++blocksIdx;
        ++heightfieldIdx;
        ++layersIdx;
        ++streamIdx;

        chunkPtr->setState(ChunkState::FILLED);
        chunkPtr->setNotReadyForQueue();

        checkChunkAndNeighborsForNeedsVbos(chunkPtr);
        for (const auto& neighborChunkPtr : chunkPtr->neighbors)
        {
            checkChunkAndNeighborsForNeedsVbos(neighborChunkPtr);
        }

        actionTimeLeft -= actionTimeFill;
    }

    while (!chunksToGatherFeaturePlacements.empty() && actionTimeLeft >= actionTimeGatherFeaturePlacements)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGatherFeaturePlacements.front();
        chunksToGatherFeaturePlacements.pop();

        chunkPtr->gatherFeaturePlacements(); // can set state to READY_TO_FILL

        actionTimeLeft -= actionTimeGatherFeaturePlacements;
    }

    while (!zonesToErode.empty() && actionTimeLeft >= actionTimeErodeZone)
    {
        needsUpdateChunks = true;

        auto zonePtr = zonesToErode.front();
        zonesToErode.pop();

        Chunk::erodeZone(
            zonePtr,
            dev_gatheredLayers[gatheredLayersIdx],
            dev_accumulatedHeights[gatheredLayersIdx],
            streams[streamIdx]
        );
        ++gatheredLayersIdx;
        ++streamIdx;

        actionTimeLeft -= actionTimeErodeZone;
    }

    while (!chunksToGenerateLayers.empty() && actionTimeLeft >= actionTimeGenerateLayers)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGenerateLayers.front();
        chunksToGenerateLayers.pop();

        chunkPtr->generateLayers(
            dev_heightfields[heightfieldIdx],
            dev_layers[layersIdx],
            dev_biomeWeights[heightfieldIdx],
            streams[streamIdx]
        );
        ++heightfieldIdx;
        ++layersIdx;
        ++streamIdx;

        addZonesToTryErosionSet(chunkPtr);

        chunkPtr->setState(ChunkState::HAS_LAYERS);

        actionTimeLeft -= actionTimeGenerateLayers;
    }

    while (!chunksToGatherHeightfield.empty() && actionTimeLeft >= actionTimeGatherHeightfield)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGatherHeightfield.front();
        chunksToGatherHeightfield.pop();

        chunkPtr->gatherHeightfield(); // can set state to NEEDS_LAYERS

        actionTimeLeft -= actionTimeGatherHeightfield;
    }

    while (!chunksToGenerateHeightfield.empty() && actionTimeLeft >= actionTimeGenerateHeightfield)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGenerateHeightfield.front();
        chunksToGenerateHeightfield.pop();

        chunkPtr->generateHeightfield(
            dev_heightfields[heightfieldIdx],
            dev_biomeWeights[heightfieldIdx],
            streams[streamIdx]
        );
        ++heightfieldIdx;
        ++streamIdx;

        chunkPtr->setState(ChunkState::HAS_HEIGHTFIELD);

        actionTimeLeft -= actionTimeGenerateHeightfield;
    }

    if (streamIdx > 0)
    {
        cudaDeviceSynchronize();
    }

#if DEBUG_TIME_CHUNK_FILL
    if (!finishedTiming)
    {
        bool areQueuesEmpty = chunksToGenerateHeightfield.empty() && chunksToGenerateLayers.empty() && chunksToGatherFeaturePlacements.empty()
            && chunksToFill.empty() && chunksToCreateAndBufferVbos.empty();

        if (!startedTiming && !areQueuesEmpty)
        {
            startedTiming = true;
            start = std::chrono::system_clock::now();
        }
        else if (startedTiming && areQueuesEmpty)
        {
            finishedTiming = true;
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsedSeconds = end - start;
            std::cout << "elapsed seconds: " << elapsedSeconds.count() << "s" << std::endl;
        }
    }
#endif
}

static const std::array<ivec3, 8> chunkCornerOffsets = {
    ivec3(0, 0, 0), ivec3(16, 0, 0), ivec3(16, 0, 16), ivec3(0, 0, 16),
    ivec3(0, 256, 0), ivec3(16, 256, 0), ivec3(16, 256, 16), ivec3(0, 256, 16)
};

void Terrain::draw(const ShaderProgram& prog, const Player* player)
{
    mat4 modelMat = mat4(1);

    for (const auto& chunkPtr : drawableChunks)
    {
        const ivec2 dist = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
        if (max(dist.x, dist.y) > chunkVbosGenRadius)
        {
            chunksToDestroyVbos.push(chunkPtr);
            continue;
        }

        if (player != nullptr)
        {
            const auto& camPos = player->getPos();
            const auto& camForward = player->getForward();

            bool shouldCull = true;
            for (int i = 0; i < 8; ++i)
            {
                const vec3 chunkCorner = chunkPtr->worldBlockPos + chunkCornerOffsets[i];
                if (glm::dot(chunkCorner - camPos, camForward) > 0)
                {
                    shouldCull = false;
                    break;
                }
            }

            if (shouldCull)
            {
                continue;
            }
        }

        modelMat[3][0] = (float)chunkPtr->worldBlockPos.x;
        modelMat[3][2] = (float)chunkPtr->worldBlockPos.z;
        prog.setModelMat(modelMat);
        prog.draw(*chunkPtr);
    }
}

ivec2 Terrain::getCurrentChunkPos() const
{
    return this->currentChunkPos;
}

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunk)
{
    this->currentChunkPos = newCurrentChunk;
}

Chunk* Terrain::debugGetCurrentChunk()
{
    ivec2 zonePos = zonePosFromChunkPos(currentChunkPos);
    const auto& zonePtr = zones[zonePos];
    return zonePtr->chunks[localChunkPosToIdx(currentChunkPos - zonePtr->worldChunkPos)].get();
}

void Terrain::debugPrintCurrentChunkState()
{
    const auto chunkPtr = debugGetCurrentChunk();
    bool isInDrawableChunks = drawableChunks.find(chunkPtr) != drawableChunks.end();

    printf("chunk (%d, %d) state: %d\n", currentChunkPos.x, currentChunkPos.y, (int)chunkPtr->getState());
    printf("is in drawable chunks: %s\n", isInDrawableChunks ? "yes" : "no");
    printf("idx count: %d\n", chunkPtr->getIdxCount());
}

void Terrain::debugPrintCurrentColumnLayers(vec2 playerPos)
{
    const auto chunkPtr = debugGetCurrentChunk();
    ivec2 blockPos = ivec2(floor(playerPos)) - ivec2(chunkPtr->worldBlockPos.x, chunkPtr->worldBlockPos.z);
    int idx = blockPos.x + 16 * blockPos.y;
    const auto& layers = chunkPtr->layers[idx];
    for (int i = 0; i < numMaterials; ++i)
    {
        printf("%s_%s_%02d: %7.3f\n", i < numStratifiedMaterials ? "s" : "e", i < numForwardMaterials ? "F" : "B", i, layers[i]);
    }
    printf("------------\n");
    printf("hgt: %7.3f\n\n", chunkPtr->heightfield[idx]);
}
