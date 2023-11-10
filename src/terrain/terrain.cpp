#include "terrain.hpp"

#include "util/enums.hpp"
#include "cuda/cudaUtils.hpp"
#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <glm/gtx/string_cast.hpp>
#include "defines.hpp"

static constexpr int chunkVbosGenRadius = 20;
static constexpr int chunkMaxGenRadius = chunkVbosGenRadius + (ZONE_SIZE * 2);

// TODO: get better estimates for these
// ================================================================================
static constexpr int totalActionTime = 500;
// ================================================================================
static constexpr int actionTimeGenerateHeightfield        = 5;
static constexpr int actionTimeGatherHeightfield          = 2;
static constexpr int actionTimeGenerateLayers             = 10;
static constexpr int actionTimeErodeZone                  = 500;
static constexpr int actionTimeGenerateFeaturePlacements  = 2;
static constexpr int actionTimeGatherFeaturePlacements    = 4;
static constexpr int actionTimeFill                       = 5;
static constexpr int actionTimeCreateAndBufferVbos        = 100;
// ================================================================================

Terrain::Terrain()
{
    initCuda();
    generateSpiral();
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
static constexpr int numDevChunkWorldBlockPositions = totalActionTime / min(actionTimeGenerateHeightfield, actionTimeGenerateLayers);
static constexpr int numDevLayers = totalActionTime / min(actionTimeGenerateLayers, actionTimeFill);
static constexpr int numDevGatheredLayers = totalActionTime / actionTimeErodeZone;
static constexpr int numStreams = max(max(numDevBlocks, numDevHeightfields), max(numDevLayers, numDevGatheredLayers)); // TODO: revisit

static Block* dev_blocks;
static FeaturePlacement* dev_featurePlacements;

static float* dev_heightfields;
static float* dev_biomeWeights; // TODO: may need to set numDevBiomeWeights = max(numDevHeightfields, numDevLayers)
                                // probably fine for now since biome weights are always used in tandem with heightfield
static ivec2* dev_chunkWorldBlockPositions;

static float* dev_layers;

static float* dev_gatheredLayers;
static float* dev_accumulatedHeights;

static std::array<cudaStream_t, numStreams> streams;

void Terrain::initCuda()
{
    cudaMalloc((void**)&dev_blocks, numDevBlocks * devBlocksSize * sizeof(Block));
    cudaMalloc((void**)&dev_featurePlacements, numDevBlocks * devFeaturePlacementsSize * sizeof(FeaturePlacement));

    cudaMalloc((void**)&dev_heightfields, numDevHeightfields * devHeightfieldSize * sizeof(float));
    cudaMalloc((void**)&dev_biomeWeights, numDevHeightfields * devBiomeWeightsSize * sizeof(float));
    cudaMalloc((void**)&dev_chunkWorldBlockPositions, numDevChunkWorldBlockPositions * sizeof(ivec2));

    cudaMalloc((void**)&dev_layers, numDevLayers * devLayersSize * sizeof(float));

    cudaMalloc((void**)&dev_gatheredLayers, numDevGatheredLayers * devGatheredLayersSize * sizeof(float));
    cudaMalloc((void**)&dev_accumulatedHeights, numDevGatheredLayers * devAccumulatedHeightsSize * sizeof(float));

    CudaUtils::checkCUDAError("cudaMalloc failed");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    CudaUtils::checkCUDAError("cudaStreamCreate failed");
}

void Terrain::freeCuda()
{
    cudaFree(dev_blocks);
    cudaFree(dev_featurePlacements);

    cudaFree(dev_heightfields);
    cudaFree(dev_biomeWeights);
    cudaFree(dev_chunkWorldBlockPositions);

    cudaFree(dev_layers);

    cudaFree(dev_gatheredLayers);
    cudaFree(dev_accumulatedHeights);

    CudaUtils::checkCUDAError("cudaFree failed");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    CudaUtils::checkCUDAError("cudaStreamDestroy failed");
}

void Terrain::generateSpiral()
{
    int spiralSideLength = chunkVbosGenRadius * 2 + 1;
    spiral.reserve(spiralSideLength * spiralSideLength);

    int x = 0;
    int z = 0;
    int d = 1;
    int m = 1;

    while (true)
    {
        while (2 * x * d < m)
        {
            spiral.push_back(ivec2(x, z));
            x += d;
        }

        if (m > chunkMaxGenRadius * 2)
        {
            return;
        }

        while (2 * z * d < m)
        {
            spiral.push_back(ivec2(x, z));
            z += d;
        }

        d = -d;
        m++;
    }
}

ivec2 chunkPosFromPlayerPos(vec2 playerPos)
{
    return ivec2(glm::floor(playerPos / 16.f));
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
    Zone* zonePtr;
    if (lastUpdateZonePtr != nullptr && newZoneWorldChunkPos == lastUpdateZonePtr->worldChunkPos)
    {
        zonePtr = lastUpdateZonePtr;
    }
    else
    {
        if (zoneIt == zones.end())
        {
            zonePtr = createZone(newZoneWorldChunkPos);
        }
        else
        {
            zonePtr = zoneIt->second.get();
        }

        this->lastUpdateZonePtr = zonePtr;
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

    const ivec2 distVec = abs(chunkPtr->worldChunkPos - this->currentChunkPos);
    int dist = max(distVec.x, distVec.y);

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
    case ChunkState::NEEDS_FEATURE_PLACEMENTS:
        chunkPtr->setNotReadyForQueue();
        chunksToGenerateFeaturePlacements.push(chunkPtr);
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

    if (dist > chunkVbosGenRadius)
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
    for (const auto& dxz : spiral)
    {
        updateChunk(dxz.x, dxz.y);
    }
}

void Terrain::addZonesToTryErosionSet(Chunk* chunkPtr)
{
    Zone* zonePtr = chunkPtr->zonePtr;
    zonesToTryErosion.insert(zonePtr); // not possible for this to already have been queued for erosion since this chunk just became ready

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
    zonePtr->gatheredChunks.resize(ZONE_SIZE * ZONE_SIZE * 4);

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
    int chunkWorldBlockPositionIdx = 0;
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
            dev_blocks + (blocksIdx * devBlocksSize),
            dev_heightfields + (heightfieldIdx * devHeightfieldSize),
            dev_biomeWeights + (heightfieldIdx * devBiomeWeightsSize),
            dev_layers + (layersIdx * devLayersSize),
            dev_featurePlacements + (blocksIdx * devFeaturePlacementsSize),
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

    while (!chunksToGenerateFeaturePlacements.empty() && actionTimeLeft >= actionTimeGenerateFeaturePlacements)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGenerateFeaturePlacements.front();
        chunksToGenerateFeaturePlacements.pop();

        chunkPtr->generateFeaturePlacements();
        chunkPtr->setState(ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS);

        actionTimeLeft -= actionTimeGenerateFeaturePlacements;
    }

    while (!zonesToErode.empty() && actionTimeLeft >= actionTimeErodeZone)
    {
        needsUpdateChunks = true;

        auto zonePtr = zonesToErode.front();
        zonesToErode.pop();

        Chunk::erodeZone(
            zonePtr,
            dev_gatheredLayers + (gatheredLayersIdx * devGatheredLayersSize),
            dev_accumulatedHeights + (gatheredLayersIdx * devAccumulatedHeightsSize),
            streams[streamIdx]
        );
        ++gatheredLayersIdx;
        ++streamIdx;

        for (const auto& chunkPtr : zonePtr->chunks)
        {
            chunkPtr->setState(ChunkState::NEEDS_FEATURE_PLACEMENTS);
        }

        actionTimeLeft -= actionTimeErodeZone;
    }

    {
        std::vector<Chunk*> chunks;
        while (!chunksToGenerateLayers.empty() && actionTimeLeft >= actionTimeGenerateLayers)
        {
            needsUpdateChunks = true;

            auto chunkPtr = chunksToGenerateLayers.front();
            chunksToGenerateLayers.pop();

            chunks.push_back(chunkPtr);

            addZonesToTryErosionSet(chunkPtr);

            chunkPtr->setState(ChunkState::HAS_LAYERS);

            actionTimeLeft -= actionTimeGenerateLayers;
        }

        int numChunks = chunks.size();
        if (numChunks > 0)
        {
            Chunk::generateLayers(
                chunks,
                dev_heightfields + (heightfieldIdx * devHeightfieldSize),
                dev_biomeWeights + (heightfieldIdx * devBiomeWeightsSize),
                dev_chunkWorldBlockPositions + (chunkWorldBlockPositionIdx),
                dev_layers + (layersIdx * devLayersSize),
                streams[streamIdx]
            );

            heightfieldIdx += numChunks;
            chunkWorldBlockPositionIdx += numChunks;
            layersIdx += numChunks;
            ++streamIdx;
        }
    }

    while (!chunksToGatherHeightfield.empty() && actionTimeLeft >= actionTimeGatherHeightfield)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGatherHeightfield.front();
        chunksToGatherHeightfield.pop();

        chunkPtr->gatherHeightfield(); // can set state to NEEDS_LAYERS

        actionTimeLeft -= actionTimeGatherHeightfield;
    }

    {
        std::vector<Chunk*> chunks;
        while (!chunksToGenerateHeightfield.empty() && actionTimeLeft >= actionTimeGenerateHeightfield)
        {
            needsUpdateChunks = true;

            auto chunkPtr = chunksToGenerateHeightfield.front();
            chunksToGenerateHeightfield.pop();

            chunks.push_back(chunkPtr);

            chunkPtr->setState(ChunkState::HAS_HEIGHTFIELD);

            actionTimeLeft -= actionTimeGenerateHeightfield;
        }

        int numChunks = chunks.size();
        if (numChunks > 0)
        {
            Chunk::generateHeightfields(
                chunks,
                dev_chunkWorldBlockPositions + (chunkWorldBlockPositionIdx),
                dev_heightfields + (heightfieldIdx * devHeightfieldSize),
                dev_biomeWeights + (heightfieldIdx * devBiomeWeightsSize),
                streams[streamIdx]
            );

            heightfieldIdx += numChunks;
            chunkWorldBlockPositionIdx += numChunks;
            ++streamIdx;
        }
    }

    if (streamIdx > 0)
    {
        cudaDeviceSynchronize();
    }

#if DEBUG_TIME_CHUNK_FILL
    if (!finishedTiming)
    {
        bool areQueuesEmpty = chunksToGenerateHeightfield.empty() && chunksToGenerateLayers.empty() && zonesToTryErosion.empty()
            && zonesToErode.empty() && chunksToGenerateFeaturePlacements.empty() && chunksToGatherFeaturePlacements.empty()
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

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunkPos)
{
    this->currentChunkPos = newCurrentChunkPos;
}

void Terrain::debugGetCurrentChunkAndZone(vec2 playerPos, Chunk** chunkPtr, Zone** zonePtr)
{
    ivec2 chunkPos = chunkPosFromPlayerPos(playerPos);
    ivec2 zonePos = zonePosFromChunkPos(chunkPos);

    const auto& zoneUptr = zones[zonePos];
    const auto& chunkUptr = zoneUptr->chunks[localChunkPosToIdx(chunkPos - zoneUptr->worldChunkPos)];

    *chunkPtr = chunkUptr.get();
    *zonePtr = zoneUptr.get();
}

void Terrain::debugPrintCurrentChunkState(vec2 playerPos)
{
    Chunk* chunkPtr;
    Zone* zonePtr;
    debugGetCurrentChunkAndZone(playerPos, &chunkPtr, &zonePtr);
    bool isInDrawableChunks = drawableChunks.find(chunkPtr) != drawableChunks.end();

    printf("===========================================================\n");
    printf("chunk (%d, %d)\n", chunkPtr->worldChunkPos.x, chunkPtr->worldChunkPos.y);
    printf("-----------------------------------------------------------\n");
    printf("chunk state: %d\n", (int)chunkPtr->getState());
    printf("chunk ready for queue: %s\n", chunkPtr->isReadyForQueue() ? "yes" : "no");
    printf("chunk in drawable chunks: %s\n", isInDrawableChunks ? "yes" : "no");
    printf("chunk idx count: %d\n", chunkPtr->getIdxCount());
    printf("===========================================================\n");
    printf("zone (%d, %d)\n", zonePtr->worldChunkPos.x, zonePtr->worldChunkPos.y);
    printf("-----------------------------------------------------------\n");
    printf("zone is ready for erosion: %s\n", isZoneReadyForErosion(zonePtr) ? "yes" : "no");
    printf("zone has been queued for erosion: %s\n", zonePtr->hasBeenQueuedForErosion ? "yes" : "no");
    printf("===========================================================\n");
    printf("\n");
}

void Terrain::debugPrintCurrentColumnLayers(vec2 playerPos)
{
    Chunk* chunkPtr;
    Zone* zonePtr;
    debugGetCurrentChunkAndZone(playerPos, &chunkPtr, &zonePtr);
    ivec2 blockPos = ivec2(floor(playerPos)) - ivec2(chunkPtr->worldBlockPos.x, chunkPtr->worldBlockPos.z);
    int idx2d = blockPos.x + 16 * blockPos.y;
    for (int layerIdx = 0; layerIdx < numMaterials; ++layerIdx)
    {
        printf(
            "%s_%s_%02d: %7.3f\n",
            layerIdx < numStratifiedMaterials ? "s" : "e",
            layerIdx < numForwardMaterials ? "F" : "B",
            layerIdx,
            chunkPtr->layers[idx2d + (256 * layerIdx)]
        );
    }
    printf("------------\n");
    printf("hgt: %7.3f\n\n", chunkPtr->heightfield[idx2d]);
}
