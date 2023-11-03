#include "terrain.hpp"

#include "util/enums.hpp"
#include "cuda/cudaUtils.hpp"
#include <thread>
#include <iostream>
#include <chrono>
#include <glm/gtx/string_cast.hpp>

#define DEBUG_TIME_CHUNK_FILL 0

static constexpr int chunkVbosGenRadius = 12;
// [+1] Gather heightfields of 3x3 chunks and place material layers
// [+1] Gather material layers of 2x2 chunks (3x3 with closer half or quarter of neighbors) to do erosion
// [+2] Gather eroded material layers and feature placements of 5x5 chunks and fill features
// [+2] Extra padding to minimize VBO recreation
static constexpr int chunkMaxGenRadius = chunkVbosGenRadius + 6;

// TODO: get better estimates for these
// ================================================================================
static constexpr int totalActionTime = 100;
// ================================================================================
static constexpr int actionTimeGenerateHeightfield        = 4;
static constexpr int actionTimeGenerateLayers             = 6;
static constexpr int actionTimeGatherFeaturePlacements    = 2;
static constexpr int actionTimeFill                       = 4;
static constexpr int actionTimeCreateVbos                 = 20;
static constexpr int actionTimeBufferVbos                 = totalActionTime / 5;
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
static constexpr int numStreams = max(numDevBlocks, max(numDevHeightfields, numDevLayers));

static std::array<Block*, numDevBlocks> dev_blocks;
static std::array<FeaturePlacement*, numDevBlocks> dev_featurePlacements;

static std::array<float*, numDevHeightfields> dev_heightfields;
static std::array<float*, numDevLayers> dev_layers;
static std::array<float*, numDevHeightfields> dev_biomeWeights; // TODO: may need to set numDevBiomeWeights = max(numDevHeightfields, numDevLayers)
                                                                // probably fine for now since biome weights are always used in tandem with heightfield
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
        cudaMalloc((void**)&dev_heightfields[i], 256 * sizeof(float));
        cudaMalloc((void**)&dev_biomeWeights[i], 256 * (int)Biome::numBiomes * sizeof(float));
    }

    for (int i = 0; i < numDevLayers; ++i)
    {
        cudaMalloc((void**)&dev_layers[i], 256 * (int)Material::numMaterials * sizeof(float));
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

    CudaUtils::checkCUDAError("cudaFree failed");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    CudaUtils::checkCUDAError("cudaStreamDestroy failed");
}

ivec2 zonePosFromChunkPos(ivec2 chunkPos)
{
    return ivec2(glm::floor(vec2(chunkPos) / 16.f)) * 16;
}

int localChunkPosToIdx(ivec2 localChunkPos)
{
    return localChunkPos.x + 16 * localChunkPos.y;
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
        return;
    }

    switch (chunkPtr->getState())
    {
    case ChunkState::EMPTY:
        chunkPtr->setNotReadyForQueue();
        chunksToGenerateHeightfield.push(chunkPtr);
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
    case ChunkState::IS_FILLED:
        chunkPtr->setNotReadyForQueue();
        chunksToCreateVbos.push(chunkPtr);
        return;
    }
}

// This function looks at the area covered by CHUNK_GEN_HEIGHTFIELDS_RADIUS. For each chunk, it first
// creates the chunk if it doesn't exist, also creating the associated zone if necsesary. Then, it looks at
// the chunk's state and adds it to the correct queue iff chunk.readyForQueue == true. This also includes
// setting readyForQueue to false so the chunk will be skipped until it's ready for the next state. The states 
// themselves, as well as readyForQueue, will be updated after the associated call or CPU thread finishes 
// execution. Kernels and threads are spawned from Terrain::tick().
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

void Terrain::createChunkVbos(Chunk* chunkPtr)
{
    chunkPtr->createVBOs();
    chunkPtr->setNotReadyForQueue();
    chunksToBufferVbos.push(chunkPtr);
}

void Terrain::tick()
{
    while (!chunksToDestroyVbos.empty())
    {
        auto chunkPtr = chunksToDestroyVbos.front();
        chunksToDestroyVbos.pop();

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

    int actionTimeLeft = totalActionTime;

    int blocksIdx = 0;
    int heightfieldIdx = 0;
    int layersIdx = 0;
    int streamIdx = 0;

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

        chunkPtr->setState(ChunkState::NEEDS_LAYERS);

        actionTimeLeft -= actionTimeGenerateHeightfield;
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

        chunkPtr->setState(ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS);

        actionTimeLeft -= actionTimeGenerateLayers;
    }

    while (!chunksToGatherFeaturePlacements.empty() && actionTimeLeft >= actionTimeGatherFeaturePlacements)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToGatherFeaturePlacements.front();
        chunksToGatherFeaturePlacements.pop();

        chunkPtr->gatherFeaturePlacements(); // this will set state to READY_TO_FILL if 5x5 neighborhood chunks all have feature placements

        actionTimeLeft -= actionTimeGatherFeaturePlacements;
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

        chunkPtr->setState(ChunkState::IS_FILLED);

        for (int i = 0; i < 4; ++i)
        {
            Chunk* neighborChunkPtr = chunkPtr->neighbors[i];
            if (neighborChunkPtr != nullptr && neighborChunkPtr->getState() == ChunkState::DRAWABLE)
            {
                chunksToDestroyVbos.push(neighborChunkPtr);
            }
        }

        actionTimeLeft -= actionTimeFill;
    }

    if (streamIdx > 0)
    {
        cudaDeviceSynchronize();
    }

    while (!chunksToCreateVbos.empty() && actionTimeLeft >= actionTimeCreateVbos)
    {
        needsUpdateChunks = true;

        auto chunkPtr = chunksToCreateVbos.front();
        chunksToCreateVbos.pop();

        this->createChunkVbos(chunkPtr);

        actionTimeLeft -= actionTimeCreateVbos;
    }

    while (!chunksToBufferVbos.empty() && actionTimeLeft >= actionTimeBufferVbos)
    {
        auto chunkPtr = chunksToBufferVbos.front();
        chunksToBufferVbos.pop();

        chunkPtr->bufferVBOs();
        drawableChunks.insert(chunkPtr);
        chunkPtr->setState(ChunkState::DRAWABLE);

        actionTimeLeft -= actionTimeBufferVbos;
    }

#if DEBUG_TIME_CHUNK_FILL
    if (!finishedTiming)
    {
        bool areQueuesEmpty = chunksToGenerateHeightfield.empty() && chunksToGenerateLayers.empty() && chunksToGatherFeaturePlacements.empty()
            && chunksToFill.empty() && chunksToCreateVbos.empty() && chunksToBufferVbos.empty();

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

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunk)
{
    this->currentChunkPos = newCurrentChunk;
}
