#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "featurePlacement.hpp"
#include "util/rng.hpp"
#include "defines.hpp"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define DEBUG_SKIP_EROSION 0
#define DEBUG_USE_CONTRIBUTION_FILL_METHOD 0

//#define DEBUG_BIOME_OVERRIDE Biome::ICEBERGS
//#define DEBUG_CAVE_BIOME_OVERRIDE CaveBiome::NONE

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos), worldBlockPos(worldChunkPos.x * 16, 0, worldChunkPos.y * 16)
{}

#pragma region state functions

ChunkState Chunk::getState() const
{
    return this->state;
}

void Chunk::setState(ChunkState newState)
{
    this->state = newState;
    this->readyForQueue = true;
}

bool Chunk::isReadyForQueue()
{
    return this->readyForQueue;
}

void Chunk::setNotReadyForQueue()
{
    this->readyForQueue = false;
}

#pragma endregion

#pragma region flood fill and iterate

// Flood fill neighborChunks (connected chunks that exist and are at or past minState).
// If a chunk's neighbor area is ready to go, it will be reached by flood fill (since this chunk is contained in that area).
// By the contrapositive, if a chunk was not reached by flood fill, its neighbor area is not ready to go.
template<std::size_t diameter>
void Chunk::floodFill(Chunk* (&neighborChunks)[diameter][diameter], ChunkState minState)
{
    constexpr int radius = diameter / 2;

    std::queue<Chunk*> chunks;
    std::unordered_set<Chunk*> visitedChunks;
    chunks.push(this);

    while (!chunks.empty())
    {
        auto chunkPtr = chunks.front();
        chunks.pop();
        visitedChunks.insert(chunkPtr);

        if (chunkPtr->getState() < minState)
        {
            continue;
        }

        ivec2 neighborChunksIdx = chunkPtr->worldChunkPos - this->worldChunkPos + ivec2(radius, radius);
        neighborChunks[neighborChunksIdx.y][neighborChunksIdx.x] = chunkPtr;

        for (const auto& neighborPtr : chunkPtr->neighbors)
        {
            if (neighborPtr == nullptr || visitedChunks.find(neighborPtr) != visitedChunks.end())
            {
                continue;
            }

            const ivec2 dist = abs(neighborPtr->worldChunkPos - this->worldChunkPos);
            if (max(dist.x, dist.y) > radius)
            {
                continue;
            }

            chunks.push(neighborPtr);
        }
    }
}

template<std::size_t diameter>
void Chunk::iterateNeighborChunks(Chunk* const (&neighborChunks)[diameter][diameter], ChunkState currentState, ChunkState nextState,
    ChunkProcessorFunc<diameter> chunkProcessorFunc)
{
    constexpr int start = diameter / 4; // assuming diameter = (4k + 1) for some k, so start <- k
    constexpr int end = diameter - start;

    for (int centerZ = start; centerZ < end; ++centerZ)
    {
        for (int centerX = start; centerX < end; ++centerX)
        {
            const auto& chunkPtr = neighborChunks[centerZ][centerX];

            if (chunkPtr == nullptr || chunkPtr->getState() != currentState)
            {
                continue;
            }

            bool isReady = true;
            for (int offsetZ = -start; offsetZ <= start && isReady; ++offsetZ)
            {
                for (int offsetX = -start; offsetX <= start && isReady; ++offsetX)
                {
                    if (neighborChunks[centerZ + offsetZ][centerX + offsetX] == nullptr)
                    {
                        isReady = false;
                        break;
                    }
                }

                if (!isReady)
                {
                    break;
                }
            }

            if (isReady)
            {
                chunkProcessorFunc(chunkPtr, neighborChunks, centerX, centerZ);
                chunkPtr->setState(nextState);
            }
        }
    }
}

template<std::size_t diameter>
void Chunk::floodFillAndIterateNeighbors(ChunkState currentState, ChunkState nextState, ChunkProcessorFunc<diameter> chunkProcessorFunc)
{
    Chunk* neighborChunks[diameter][diameter] = {};
    floodFill<diameter>(neighborChunks, currentState);
    iterateNeighborChunks<diameter>(neighborChunks, currentState, nextState, chunkProcessorFunc);
}

#pragma endregion

#pragma region heightfield

__global__ void kernGenerateHeightfield(
    ivec2* chunkWorldBlockPositions,
    float* heightfield,
    float* biomeWeights)
{
    const int chunkIdx = blockIdx.x * blockDim.x;

    const int x = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posTo2dIndex(x, z);

    const vec2 worldPos = chunkWorldBlockPositions[chunkIdx] + ivec2(x, z);
    const auto biomeNoise = getBiomeNoise(worldPos);

    float* columnBiomeWeights = biomeWeights + (devBiomeWeightsSize * chunkIdx) + (idx);
    float height = 0.f;
    for (int biomeIdx = 0; biomeIdx < numBiomes; ++biomeIdx)
    {
        Biome biome = (Biome)biomeIdx;

#ifdef DEBUG_BIOME_OVERRIDE
        float weight = (biome == DEBUG_BIOME_OVERRIDE) ? 1.f : 0.f;
#else
        float weight = getBiomeWeight(biome, biomeNoise);
#endif
        if (weight > 0.f)
        {
            height += weight * getHeight(biome, worldPos);
        }

        columnBiomeWeights[256 * biomeIdx] = weight;
    }

    heightfield[(256 * chunkIdx) + idx] = height;
}

void Chunk::generateHeightfields(
    std::vector<Chunk*>& chunks,
    ivec2* host_chunkWorldBlockPositions,
    ivec2* dev_chunkWorldBlockPositions,
    float* host_heightfields,
    float* dev_heightfields,
    float* host_biomeWeights,
    float* dev_biomeWeights,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    for (int i = 0; i < numChunks; ++i)
    {
        ivec3 worldBlockPos = chunks[i]->worldBlockPos;
        host_chunkWorldBlockPositions[i] = ivec2(worldBlockPos.x, worldBlockPos.z);
    }

    cudaMemcpyAsync(dev_chunkWorldBlockPositions, host_chunkWorldBlockPositions, numChunks * sizeof(ivec2), cudaMemcpyHostToDevice, stream);

    const dim3 blockSize3d(1, 16, 16);
    const dim3 blocksPerGrid3d(numChunks, 1, 1);
    kernGenerateHeightfield<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
        dev_chunkWorldBlockPositions,
        dev_heightfields,
        dev_biomeWeights
    );

    cudaMemcpyAsync(host_heightfields, dev_heightfields, numChunks * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_biomeWeights, dev_biomeWeights, numChunks * devBiomeWeightsSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
  
    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->heightfield.data(), host_heightfields + (256 * i), 256 * sizeof(float));
        std::memcpy(chunkPtr->biomeWeights.data(), host_biomeWeights + (devBiomeWeightsSize * i), devBiomeWeightsSize * sizeof(float));
    }

    CudaUtils::checkCUDAError("Chunk::generateHeightfield() failed");
}

void calculateEdgeIndices(int offset, int& in, int& out)
{
    in = (offset == -1) ? 15 : 0;
    out = (offset == -1) ? 0 : 17;
}

void Chunk::otherChunkGatherHeightfield(Chunk* chunkPtr, Chunk* const (&neighborChunks)[5][5], int centerX, int centerZ)
{
    chunkPtr->gatheredHeightfield.resize(18 * 18);

    for (const auto& neighborDir : DirectionEnums::dirVecs2d)
    {
        int offsetX = neighborDir[0];
        int offsetZ = neighborDir[1];

        const auto& neighborPtr = neighborChunks[centerZ + offsetZ][centerX + offsetX];

        if (offsetX == 0 || offsetZ == 0)
        {
            // edge
            if (offsetZ == 0)
            {
                // +/- localX
                int xIn, xOut;
                calculateEdgeIndices(offsetX, xIn, xOut);

                for (int z = 0; z < 16; ++z)
                {
                    chunkPtr->gatheredHeightfield[posTo2dIndex<18>(xOut, z + 1)] = neighborPtr->heightfield[posTo2dIndex(xIn, z)];
                }
            }
            else
            {
                // +/- localZ
                int zIn, zOut;
                calculateEdgeIndices(offsetZ, zIn, zOut);

                for (int x = 0; x < 16; ++x)
                {
                    chunkPtr->gatheredHeightfield[posTo2dIndex<18>(x + 1, zOut)] = neighborPtr->heightfield[posTo2dIndex(x, zIn)];
                }
            }
        }
        else
        {
            // corner
            int xIn, xOut, zIn, zOut;
            calculateEdgeIndices(offsetX, xIn, xOut);
            calculateEdgeIndices(offsetZ, zIn, zOut);
            chunkPtr->gatheredHeightfield[posTo2dIndex<18>(xOut, zOut)] = neighborPtr->heightfield[posTo2dIndex(xIn, zIn)];
        }
    }

    // copy chunk's own heightfield into gathered heightfield
    for (int z = 0; z < 16; ++z)
    {
        std::memcpy(
            chunkPtr->gatheredHeightfield.data() + posTo2dIndex<18>(1, z + 1),
            chunkPtr->heightfield.data() + (16 * z),
            16 * sizeof(float)
        );
    }
}

void Chunk::gatherHeightfield()
{
    floodFillAndIterateNeighbors<5>(
        ChunkState::HAS_HEIGHTFIELD,
        ChunkState::NEEDS_LAYERS,
        &Chunk::otherChunkGatherHeightfield
    );
}

#pragma endregion

#pragma region layers

__device__ float getStratifiedMaterialThickness(int layerIdx, float materialWeight, vec2 worldPos)
{
    if (materialWeight > 0)
    {
        const auto& materialInfo = dev_materialInfos[layerIdx];
        vec2 noisePos = worldPos * materialInfo.noiseScaleOrMaxSlope + vec2(layerIdx * 5283.64f);
        return max(0.f, materialInfo.thickness + materialInfo.noiseAmplitudeOrTanAngleOfRepose * fbm(noisePos)) * materialWeight;
    }
    else
    {
        return 0;
    }
}

__global__ void kernGenerateLayers(
    float* heightfield,
    float* biomeWeights,
    ivec2* chunkWorldBlockPositions,
    float* layers)
{
    __shared__ float shared_heightfield[18 * 18];

    const int chunkIdx = blockIdx.x * blockDim.x;

    const int x = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posTo2dIndex(x, z);

    const vec2 worldPos = chunkWorldBlockPositions[chunkIdx] + ivec2(x, z);

    shared_heightfield[idx] = heightfield[(devHeightfieldSize * chunkIdx) + idx];
    const int idx2 = idx + 256;
    if (idx2 < 18 * 18)
    {
        shared_heightfield[idx2] = heightfield[(devHeightfieldSize * chunkIdx) + idx2];
    }

    __syncthreads();

    float totalMaterialWeights[numMaterials];
    #pragma unroll
    for (int materialIdx = 0; materialIdx < numMaterials; ++materialIdx)
    {
        totalMaterialWeights[materialIdx] = 0;
    }

    const float* columnBiomeWeights = biomeWeights + (devBiomeWeightsSize * chunkIdx) + (idx);
    #pragma unroll
    for (int biomeIdx = 0; biomeIdx < numBiomes; ++biomeIdx)
    {
        const float biomeWeight = columnBiomeWeights[256 * biomeIdx];

        #pragma unroll
        for (int materialIdx = 0; materialIdx < numMaterials; ++materialIdx)
        {
            totalMaterialWeights[materialIdx] += biomeWeight * dev_biomeMaterialWeights[posTo2dIndex<numMaterials>(materialIdx, biomeIdx)];
        }
    }

    const ivec2 pos18 = ivec2(x + 1, z + 1);
    const float maxHeight = shared_heightfield[posTo2dIndex<18>(pos18)];

    float slope = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float neighborHeight = shared_heightfield[posTo2dIndex<18>(pos18 + dev_dirVecs2d[i])];
        slope = max(slope, abs(neighborHeight - maxHeight) * (i % 2 == 1 ? SQRT_2 : 1));
    }

    float* columnLayers = layers + (devLayersSize * chunkIdx) + (idx);

    float height = 0;
    #pragma unroll
    for (int layerIdx = 0; layerIdx < numForwardMaterials; ++layerIdx)
    {
        columnLayers[256 * layerIdx] = height;

        if (height > maxHeight || layerIdx == numForwardMaterials - 1)
        {
            break;
        }

        height += getStratifiedMaterialThickness(layerIdx, totalMaterialWeights[layerIdx], worldPos);
    }

    height = 0;
    #pragma unroll
    for (int layerIdx = numStratifiedMaterials - 1; layerIdx >= numForwardMaterials; --layerIdx)
    {
        height += getStratifiedMaterialThickness(layerIdx, totalMaterialWeights[layerIdx], worldPos);
        columnLayers[256 * layerIdx] = height; // actual height is calculated by in Chunk::fixBackwardStratifiedLayers() subtracting this value from start height of eroded layers
    }

    height = maxHeight;
    #pragma unroll
    for (int layerIdx = numMaterials - 1; layerIdx >= numStratifiedMaterials; --layerIdx)
    {
        const auto& materialInfo = dev_materialInfos[layerIdx];

        float materialWeight = totalMaterialWeights[layerIdx];
        float layerHeight = max(0.f, materialInfo.thickness * ((materialInfo.noiseScaleOrMaxSlope - slope) / materialInfo.noiseScaleOrMaxSlope)) * materialWeight;

        height -= layerHeight;
        columnLayers[256 * layerIdx] = height;
    }
}

void Chunk::generateLayers(
    std::vector<Chunk*>& chunks,
    float* host_heightfields,
    float* dev_heightfields,
    float* host_biomeWeights,
    float* dev_biomeWeights,
    ivec2* host_chunkWorldBlockPositions,
    ivec2* dev_chunkWorldBlockPositions,
    float* host_layers,
    float* dev_layers,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(host_heightfields + (i * devHeightfieldSize), chunkPtr->gatheredHeightfield.data(), devHeightfieldSize * sizeof(float));
        chunkPtr->gatheredHeightfield.clear();

        std::memcpy(host_biomeWeights + (i * devBiomeWeightsSize), chunkPtr->biomeWeights.data(), devBiomeWeightsSize * sizeof(float));

        ivec3 worldBlockPos = chunkPtr->worldBlockPos;
        host_chunkWorldBlockPositions[i] = ivec2(worldBlockPos.x, worldBlockPos.z);
    }

    cudaMemcpyAsync(dev_heightfields, host_heightfields, numChunks * devHeightfieldSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_biomeWeights, host_biomeWeights, numChunks * devBiomeWeightsSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_chunkWorldBlockPositions, host_chunkWorldBlockPositions, numChunks * sizeof(ivec2), cudaMemcpyHostToDevice, stream);

    const dim3 blockSize3d(1, 16, 16);
    const dim3 blocksPerGrid3d(numChunks, 1, 1);
    kernGenerateLayers<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
        dev_heightfields,
        dev_biomeWeights,
        dev_chunkWorldBlockPositions,
        dev_layers
    );

    cudaMemcpyAsync(host_layers, dev_layers, numChunks * devLayersSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->layers.data(), host_layers + (i * devLayersSize), devLayersSize * sizeof(float));
    }

    CudaUtils::checkCUDAError("Chunk::generateLayers() failed");
}

#pragma endregion

#pragma region erosion

static constexpr int gatheredLayersBaseSize = EROSION_GRID_NUM_COLS * (numErodedMaterials + 1); // +1 for heightfield

__global__ void kernDoErosion(
    float* gatheredLayers, 
    float* accumulatedHeights,
    int layerIdx,
    bool isFirst)
{
    __shared__ float shared_layerStart[34 * 34]; // 32x32 with 1 padding
    __shared__ float shared_layerEnd[34 * 34];
    __shared__ bool shared_didChange;

    const int localX = threadIdx.x;
    const int localZ = threadIdx.y;
    const int localIdx2d = posTo2dIndex<32>(localX, localZ);

    const int blockStartX = (blockIdx.x * blockDim.x);
    const int blockStartZ = (blockIdx.y * blockDim.y);

    const int globalX = blockStartX + localX;
    const int globalZ = blockStartZ + localZ;
    const int globalIdx2d = posTo2dIndex<EROSION_GRID_SIDE_LENGTH_BLOCKS>(globalX, globalZ);

    if (localIdx2d == 0)
    {
        shared_didChange = false;
    }

    const ivec2 sharedLayerPos = ivec2(localX + 1, localZ + 1);
    const int sharedLayerIdx = posTo2dIndex<34>(sharedLayerPos);
    const int gatheredLayersIdx = globalIdx2d + (EROSION_GRID_NUM_COLS * layerIdx);

    float thisAccumulatedHeight = isFirst ? accumulatedHeights[globalIdx2d] : 0;

    const float thisLayerStart = gatheredLayers[gatheredLayersIdx] + thisAccumulatedHeight;
    const float thisLayerEnd = gatheredLayers[gatheredLayersIdx + EROSION_GRID_NUM_COLS] + thisAccumulatedHeight;
    shared_layerStart[sharedLayerIdx] = thisLayerStart;
    shared_layerEnd[sharedLayerIdx] = thisLayerEnd;

    ivec2 storePos = ivec2(-1);
    if (localIdx2d < 64)
    {
        storePos = ivec2((localIdx2d % 32) + 1, localIdx2d < 32 ? 0 : 33);
    }
    else if (localIdx2d < 128)
    {
        storePos = ivec2(localIdx2d < 96 ? 0 : 33, (localIdx2d % 32) + 1);
    }
    else
    {
        switch (localIdx2d)
        {
        case 128:
            storePos = ivec2(0, 0);
            break;
        case 129:
            storePos = ivec2(33, 0);
            break;
        case 130:
            storePos = ivec2(0, 33);
            break;
        case 131:
            storePos = ivec2(33, 33);
            break;
        }
    }

    if (storePos.x != -1)
    {
        ivec2 loadPos = ivec2(blockStartX - 1, blockStartZ - 1) + storePos;
        loadPos = clamp(loadPos, 0, EROSION_GRID_SIDE_LENGTH_BLOCKS - 1); // values outside the grid (i.e. not in gatheredLayers) extend existing border values

        const int loadIdx2d = posTo2dIndex<EROSION_GRID_SIDE_LENGTH_BLOCKS>(loadPos);
        const int loadIdx = loadIdx2d + (EROSION_GRID_NUM_COLS * layerIdx);
        const int storeIdx = posTo2dIndex<34>(storePos);

        thisAccumulatedHeight = isFirst ? accumulatedHeights[loadIdx2d] : 0;

        shared_layerStart[storeIdx] = gatheredLayers[loadIdx] + thisAccumulatedHeight;
        shared_layerEnd[storeIdx] = gatheredLayers[loadIdx + EROSION_GRID_NUM_COLS] + thisAccumulatedHeight;
    }

    __syncthreads();

    float newLayerStart = thisLayerStart;
    float maxThickness = thisLayerEnd - thisLayerStart;
    const float tanAngleOfRepose = dev_materialInfos[numStratifiedMaterials + layerIdx].noiseAmplitudeOrTanAngleOfRepose;

    for (int i = 0; i < 8; ++i)
    {
        const auto& neighborDir = dev_dirVecs2d[i];
        int neighborIdx = posTo2dIndex<34>(sharedLayerPos + neighborDir);

        float neighborLayerStart = shared_layerStart[neighborIdx];
        newLayerStart = max(newLayerStart, neighborLayerStart - tanAngleOfRepose * (i % 2 == 1 ? SQRT_2 : 1));

        maxThickness = max(maxThickness, shared_layerEnd[neighborIdx] - neighborLayerStart);
    }

    newLayerStart = min(newLayerStart, thisLayerEnd);

    if (maxThickness > 0)
    {
        gatheredLayers[gatheredLayersIdx] = newLayerStart;

        if (newLayerStart != thisLayerStart)
        {
            shared_didChange = true; // not atomic since any thread that writes to shared_didChange will write the same value
                                     // plus I think this gets serialized anyway since they're all writing to the same bank

            accumulatedHeights[globalIdx2d] += newLayerStart - thisLayerStart;
        }
    }

    __syncthreads();

    if (localIdx2d != 0)
    {
        return;
    }

    // update global flag if block-level shared flag was set
    if (shared_didChange)
    {
        gatheredLayers[gatheredLayersBaseSize] = 1; // not atomic for same reason as above
    }
}

void copyLayers(Zone* zonePtr, float* gatheredLayers, bool toGatheredLayers)
{
    const int maxDim = toGatheredLayers ? ZONE_SIZE * 2 : ZONE_SIZE;
    const int maxLayerIdx = toGatheredLayers ? numMaterials + 1 : numMaterials;

    for (int chunkZ = 0; chunkZ < maxDim; ++chunkZ)
    {
        for (int chunkX = 0; chunkX < maxDim; ++chunkX)
        {
            Chunk* chunkPtr;
            ivec2 chunkBlockPos;
            if (toGatheredLayers)
            {
                chunkPtr = zonePtr->gatheredChunks[posTo2dIndex<ZONE_SIZE * 2>(chunkX, chunkZ)];
                chunkBlockPos = ivec2(chunkX, chunkZ) * 16;
            }
            else
            {
                chunkPtr = zonePtr->chunks[posTo2dIndex<ZONE_SIZE>(chunkX, chunkZ)].get();
                chunkBlockPos = (ivec2(chunkX, chunkZ) + ivec2(ZONE_SIZE / 2)) * 16;
            }

            for (int layerIdx = numStratifiedMaterials; layerIdx < maxLayerIdx; ++layerIdx)
            {
                for (int blockZ = 0; blockZ < 16; ++blockZ)
                {
                    const int globalBlockZ = chunkBlockPos.y + blockZ;

                    float* srcLayers;
                    if (toGatheredLayers && layerIdx == maxLayerIdx - 1)
                    {
                        srcLayers = chunkPtr->heightfield.data() + (16 * blockZ);
                    }
                    else
                    {
                        srcLayers = chunkPtr->layers.data() + (16 * blockZ) + (256 * layerIdx);
                    }

                    float* dstLayers = gatheredLayers
                        + (chunkBlockPos.x)
                        + (EROSION_GRID_SIDE_LENGTH_BLOCKS * globalBlockZ)
                        + (EROSION_GRID_NUM_COLS * (layerIdx - numStratifiedMaterials));

                    if (!toGatheredLayers)
                    {
                        std::swap(srcLayers, dstLayers);
                    }

                    std::memcpy(dstLayers, srcLayers, 16 * sizeof(float));
                }
            }
        }
    }
}

__host__ void Chunk::erodeZone(
    Zone* zonePtr,
    float* host_gatheredLayers,
    float* dev_gatheredLayers,
    float* dev_accumulatedHeights,
    cudaStream_t stream)
{
#if !DEBUG_SKIP_EROSION
    copyLayers(zonePtr, host_gatheredLayers, true);
    zonePtr->gatheredChunks.clear();

    int gatheredLayersSizeBytes = gatheredLayersBaseSize * sizeof(float);
    cudaMemcpyAsync(dev_gatheredLayers, host_gatheredLayers, gatheredLayersSizeBytes, cudaMemcpyHostToDevice, stream);

    float flagDidChange;
    float* dev_flagDidChange = dev_gatheredLayers + gatheredLayersBaseSize;

    const dim3 blockSize2d(32, 32);
    constexpr int blocksPerGrid = (ZONE_SIZE * 2 * 16) / 32; // = ZONE_SIZE but writing it out for clarity
    const dim3 blocksPerGrid2d(blocksPerGrid, blocksPerGrid);

    thrust::device_ptr<float> dev_ptr(dev_accumulatedHeights);
    thrust::fill_n(dev_ptr, EROSION_GRID_NUM_COLS, 0.f);

    for (int layerIdx = numErodedMaterials - 1; layerIdx >= 0; --layerIdx)
    {
        bool isFirst = true;

        do
        {
            flagDidChange = 0;
            cudaMemcpyAsync(dev_flagDidChange, &flagDidChange, 1 * sizeof(float), cudaMemcpyHostToDevice, stream);

            kernDoErosion<<<blocksPerGrid2d, blockSize2d, 0, stream>>>(
                dev_gatheredLayers,
                dev_accumulatedHeights,
                layerIdx,
                isFirst
            );

            cudaMemcpyAsync(&flagDidChange, dev_flagDidChange, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream);

            cudaStreamSynchronize(stream);

            isFirst = false;
        }
        while (flagDidChange != 0);
    }

    cudaMemcpyAsync(host_gatheredLayers, dev_gatheredLayers, gatheredLayersSizeBytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // all data needs to be copied back to gatheredLayers before calling copyLayers()
                                   // explicit synchronization here may not be necessary (seems to work without it) but it gives peace of mind

    copyLayers(zonePtr, host_gatheredLayers, false);
#else
    zonePtr->gatheredChunks.clear();
#endif

    for (const auto& chunkPtr : zonePtr->chunks)
    {
        chunkPtr->fixBackwardStratifiedLayers();
    }

    CudaUtils::checkCUDAError("Chunk::erodeZone() failed");
}

void Chunk::fixBackwardStratifiedLayers()
{
    std::array<float, 256> erodedStartHeights;

    for (int layerIdx = numForwardMaterials; layerIdx < numStratifiedMaterials; ++layerIdx)
    {
        const int layerIdx256 = 256 * layerIdx;

        for (int localZ = 0; localZ < 16; ++localZ)
        {
            for (int localX = 0; localX < 16; ++localX)
            {
                const int idx2d = posTo2dIndex(localX, localZ);
                float* columnLayers = this->layers.data() + idx2d;

                if (layerIdx == numForwardMaterials)
                {
                    erodedStartHeights[idx2d] = columnLayers[256 * numStratifiedMaterials];
                }

                columnLayers[layerIdx256] = erodedStartHeights[idx2d] - columnLayers[layerIdx256];
            }
        }
    }
}

#pragma endregion

#pragma region caves

__device__ bool shouldGenerateCaveAtBlock(ivec3 worldPos, float maxHeight, float oceanAndBeachWeight)
{
    if (worldPos.y == 0)
    {
        return false;
    }

    if (worldPos.y > (max((int)maxHeight, SEA_LEVEL)))
    {
        return true;
    }

    vec3 noisePos = vec3(worldPos) * 0.0050f;
    float topRatioYOffset = oceanAndBeachWeight * 50.f;
    float topHeightRatio = smoothstep(142.f, 95.f, (float)worldPos.y + topRatioYOffset);
    float bottomHeightRatio = smoothstep(5.f, 20.f, (float)worldPos.y);

    vec3 noiseOffset = fbm3From3<5>(noisePos * 0.8000f) * 1.8f;
    float caveNoise = specialCaveNoise(noisePos * vec3(1.f, 1.6f, 1.f) + noiseOffset);

    float worleyEdgeThreshold = 0.24f + 0.12f * fbm<4>(noisePos * 4.f);
    float hugeCaveNoise = smoothstep(0.2f, 0.4f, fbm<4>(noisePos * 0.0700f));
    worleyEdgeThreshold *= (1.f + 1.4f * hugeCaveNoise);
    worleyEdgeThreshold *= (topHeightRatio) * (0.3f + 0.7f * bottomHeightRatio);

    if (worleyEdgeThreshold > 0.04f && caveNoise < worleyEdgeThreshold)
    {
        return true;
    }

    vec2 ravineNoisePos = vec2(worldPos.x, worldPos.z) * 0.0015f;
    vec2 ravineWorleyOffset = 0.03f * fbm2From2<4>(ravineNoisePos * 10.f);
    vec3 ravineWorleyColor;
    float ravineWorley = worley(ravineNoisePos + ravineWorleyOffset, &ravineWorleyColor);
    const float ravineWorleyThreshold = 0.12f * (1.f - oceanAndBeachWeight);
    if (ravineWorley < ravineWorleyThreshold)
    {
        float ravineTop = 120.f + 24.f * ravineWorleyColor.x;
        float ravineRatio = 1.f - (ravineWorley / ravineWorleyThreshold);

        float ravineDepth = 60.f + 26.f * fbm<4>(ravineNoisePos * 8.f + vec2(8391.32f, 4821.39f));
        ravineDepth *= smoothstep(0.f, 0.3f, ravineRatio);

        float ravineWaveNoiseOffset = 4.f * fbm<4>(ravineNoisePos * 3.f + vec2(5129.32f, 1392.49f));
        float ravineWaveNoise = sin((ravineNoisePos.x + ravineNoisePos.y) * 15.f + ravineWaveNoiseOffset);
        ravineWaveNoise = smoothstep(0.4f, 0.6f, ravineWaveNoise);
        ravineDepth *= ravineWaveNoise;

        if (ravineDepth > 0.0001f && worldPos.y > ravineTop - ravineDepth)
        {
            return true;
        }
    }

    return false;
}

__global__ void kernGenerateCaves(
    float* heightfield,
    float* biomeWeights,
    ivec2* chunkWorldBlockPositions, 
    CaveLayer* caveLayers)
{
    __shared__ float shared_maxHeight;
    __shared__ float shared_oceanAndBeachWeight;

    __shared__ int shared_isFilled[384];
    __shared__ int shared_flipHeights[384];

    const int globalX = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    const int chunkIdx = globalX / 16;
    const int x = globalX - (chunkIdx * 16);
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx2d = posTo2dIndex(x, z);

    const ivec2 chunkWorldBlockPos2d = chunkWorldBlockPositions[chunkIdx];
    const ivec3 worldPos = ivec3(chunkWorldBlockPos2d.x + x, y, chunkWorldBlockPos2d.y + z);

    if (y == 0)
    {
        shared_oceanAndBeachWeight = 0.f;
    } else if (y == 1)
    {
        shared_maxHeight = heightfield[256 * chunkIdx + idx2d];
    }

    __syncthreads();

    if (y < numOceanAndBeachBiomes)
    {
        float biomeWeight = biomeWeights[devBiomeWeightsSize * chunkIdx + 256 * y + idx2d];
        atomicAdd(&shared_oceanAndBeachWeight, biomeWeight);
    }

    __syncthreads();

    int isThisFilled = shouldGenerateCaveAtBlock(worldPos, shared_maxHeight, shared_oceanAndBeachWeight) ? 0 : 1;
    shared_isFilled[y] = isThisFilled;

    __syncthreads();

    int isNextFilled = y < 383 ? shared_isFilled[y + 1] : 0;
    shared_flipHeights[y] = (isThisFilled != isNextFilled) ? y : -1;

    __syncthreads();

    if (y >= max(384 / 32, MAX_CAVE_LAYERS_PER_COLUMN))
    {
        return;
    }

    CaveLayer* columnCaveLayers = caveLayers + (256 * MAX_CAVE_LAYERS_PER_COLUMN * chunkIdx) + (MAX_CAVE_LAYERS_PER_COLUMN * idx2d);

    if (y < (384 / 32))
    {
        const int startLoadIdx = 32 * y;
        int endLoadIdx = startLoadIdx;
        // TODO: see if this causes bank conflicts (pretty sure it does), could be fixable by making shared_flipHeights size 33 * 12
        for (int i = startLoadIdx; i < startLoadIdx + 32; ++i)
        {
            int flipHeight = shared_flipHeights[i];
            if (flipHeight != -1)
            {
                shared_flipHeights[endLoadIdx] = flipHeight;
                ++endLoadIdx;
            }
        }

        const int numFlips = endLoadIdx - startLoadIdx;
        int startStoreIdx = 0;

        // add up numFlips from all threads with lower y and store into startStoreIdx
        // TODO: replace with parallel version
        for (int srcLane = 0; srcLane < 12; ++srcLane)
        {
            int srcLaneNumFlips = __shfl_sync(0x00000fffu, numFlips, srcLane);
            if (srcLane < y)
            {
                startStoreIdx += srcLaneNumFlips;
            }
        }

        int* columnCaveLayersInts = (int*)columnCaveLayers;

        for (int i = 0; i < numFlips; ++i)
        {
            int storeIdx = startStoreIdx + i;
            storeIdx += (storeIdx >> 1); // skip biomes and padding
            columnCaveLayersInts[storeIdx] = shared_flipHeights[startLoadIdx + i];
        }
    }

    if (y < MAX_CAVE_LAYERS_PER_COLUMN)
    {
        CaveLayer& caveLayer = columnCaveLayers[y];
        const ivec2 worldBlockPos2d = chunkWorldBlockPos2d + ivec2(x, z);

        if (caveLayer.start != 384)
        {
#ifdef DEBUG_CAVE_BIOME_OVERRIDE
            caveLayer.bottomBiome = DEBUG_CAVE_BIOME_OVERRIDE;
#else
            caveLayer.bottomBiome = getCaveBiome(ivec3(worldBlockPos2d.x, caveLayer.start, worldBlockPos2d.y), shared_maxHeight, 329271348);
#endif
        }

        if (caveLayer.end == 384)
        {
            caveLayer.topBiome = CaveBiome::NONE;
        }
        else
        {
#ifdef DEBUG_CAVE_BIOME_OVERRIDE
            caveLayer.topBiome = DEBUG_CAVE_BIOME_OVERRIDE;
#else
            caveLayer.topBiome = getCaveBiome(ivec3(worldBlockPos2d.x, caveLayer.end + 1, worldBlockPos2d.y), shared_maxHeight, 4982921);
#endif
        }
    }
}

__host__ void Chunk::generateCaves(
    std::vector<Chunk*>& chunks,
    float* host_heightfields,
    float* dev_heightfields,
    float* host_biomeWeights,
    float* dev_biomeWeights,
    ivec2* host_chunkWorldBlockPositions,
    ivec2* dev_chunkWorldBlockPositions,
    CaveLayer* host_caveLayers,
    CaveLayer* dev_caveLayers,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(host_heightfields + (i * 256), chunkPtr->heightfield.data(), 256 * sizeof(float));
        chunkPtr->gatheredHeightfield.clear();

        std::memcpy(host_biomeWeights + (i * devBiomeWeightsSize), chunkPtr->biomeWeights.data(), devBiomeWeightsSize * sizeof(float));

        ivec3 worldBlockPos = chunkPtr->worldBlockPos;
        host_chunkWorldBlockPositions[i] = ivec2(worldBlockPos.x, worldBlockPos.z);
    }

    cudaMemcpyAsync(dev_heightfields, host_heightfields, numChunks * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_biomeWeights, host_biomeWeights, numChunks * devBiomeWeightsSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_chunkWorldBlockPositions, host_chunkWorldBlockPositions, numChunks * sizeof(ivec2), cudaMemcpyHostToDevice, stream);

    thrust::device_ptr<CaveLayer> dev_ptr(dev_caveLayers);
    CaveLayer defaultCaveLayer = { 384, 384 };
    thrust::fill_n(dev_ptr, numChunks * devCaveLayersSize, defaultCaveLayer);

    const dim3 blockSize3d(1, 384, 1);
    const dim3 blocksPerGrid3d(numChunks * 16, 1, 16);
    kernGenerateCaves<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
        dev_heightfields,
        dev_biomeWeights,
        dev_chunkWorldBlockPositions,
        dev_caveLayers
    );

    cudaMemcpyAsync(host_caveLayers, dev_caveLayers, numChunks * devCaveLayersSize * sizeof(CaveLayer), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->caveLayers.data(), host_caveLayers + (i * devCaveLayersSize), devCaveLayersSize * sizeof(CaveLayer));
    }
}

#pragma endregion

#pragma region feature placements

bool isFeaturePos(ivec2 worldBlockPos2d, int gridCellSize, int gridCellPadding, int seed)
{
    const ivec2 gridCornerWorldPos = ivec2(floor(vec2(worldBlockPos2d) / (float)gridCellSize) * (float)gridCellSize);
    const int gridCellInternalSideLength = gridCellSize - (2 * gridCellPadding);
    vec2 randPos = rand2From3(vec3(gridCornerWorldPos, seed));
    const ivec2 gridPlaceWorldPos = gridCornerWorldPos
        + ivec2(gridCellPadding)
        + ivec2(floor(randPos * (float)gridCellInternalSideLength));
    return worldBlockPos2d == gridPlaceWorldPos;
}

bool Chunk::tryGenerateCaveFeaturePlacement(
    const CaveFeatureGen& caveFeatureGen,
    const CaveLayer& caveLayer,
    bool top,
    int caveFeaturePlacementSeed,
    float rand,
    ivec2 worldBlockPos2d)
{
    int layerHeight = caveLayer.end - caveLayer.start;

    if (rand >= caveFeatureGen.chancePerGridCell
        || (top != caveFeatureGen.generatesFromCeiling)
        || (!caveFeatureGen.canGenerateInLava && (top ? caveLayer.end : (caveLayer.start + 1)) <= LAVA_LEVEL)
        || layerHeight < caveFeatureGen.minLayerHeight)
    {
        return false;
    }

    if (isFeaturePos(worldBlockPos2d, caveFeatureGen.gridCellSize, caveFeatureGen.gridCellPadding, caveFeaturePlacementSeed))
    {
        this->caveFeaturePlacements.push_back({
            caveFeatureGen.caveFeature,
            ivec3(worldBlockPos2d.x, caveLayer.start + 1, worldBlockPos2d.y),
            layerHeight,
            caveFeatureGen.canReplaceBlocks
        });
        return true;
    }
}

// TODO: maybe revisit and try to optimize this by calculating thicknesses in a separate pass with better memory access patterns
void Chunk::generateColumnFeaturePlacements(int localX, int localZ)
{
    const int idx2d = posTo2dIndex(localX, localZ);

    const float* columnBiomeWeights = biomeWeights.data() + idx2d;

    const float height = heightfield[idx2d];
    const int groundHeight = (int)height;

    const ivec2 localBlockPos2d = ivec2(localX, localZ);
    const ivec2 worldBlockPos2d = ivec2(this->worldBlockPos.x, this->worldBlockPos.z) + localBlockPos2d;

    auto blockRng = makeSeededRandomEngine(worldBlockPos2d.x, worldBlockPos2d.y, 329828101);
    thrust::uniform_real_distribution<float> u01(0, 1);

    bool surfaceIsCave = false;
    const auto columnCaveLayers = this->caveLayers.data() + (idx2d * MAX_CAVE_LAYERS_PER_COLUMN);
    for (int caveLayerIdx = 0; caveLayerIdx < MAX_CAVE_LAYERS_PER_COLUMN; ++caveLayerIdx)
    {
        const auto& caveLayer = columnCaveLayers[caveLayerIdx];

        if (caveLayer.start == 384 || groundHeight <= caveLayer.start)
        {
            break;
        }

        for (const auto& caveFeatureGen : host_caveBiomeFeatureGens[(int)caveLayer.bottomBiome])
        {
            int caveFeaturePlacementSeed = (int)caveFeatureGen.caveFeature * 98239 + caveLayerIdx * 191702;
            if (tryGenerateCaveFeaturePlacement(caveFeatureGen, caveLayer, false, caveFeaturePlacementSeed, u01(blockRng), worldBlockPos2d))
            {
                break;
            }
        }

        if (caveLayer.end != 384)
        {
            for (const auto& caveFeatureGen : host_caveBiomeFeatureGens[(int)caveLayer.topBiome])
            {
                int caveFeaturePlacementSeed = (int)caveFeatureGen.caveFeature * 58321 + caveLayerIdx * 871503;
                if (tryGenerateCaveFeaturePlacement(caveFeatureGen, caveLayer, true, caveFeaturePlacementSeed, u01(blockRng), worldBlockPos2d))
                {
                    break;
                }
            }
        }

        if (groundHeight > caveLayer.start && groundHeight <= caveLayer.end)
        {
            surfaceIsCave = true;
            break;
        }
    }

    // generate surface features
    if (!surfaceIsCave)
    {
        Biome biome = getRandomBiome<256>(columnBiomeWeights, u01(blockRng));
        const auto& featureGens = host_biomeFeatureGens[(int)biome];

        const float* columnLayers = this->layers.data() + idx2d;

        for (const auto& featureGen : featureGens)
        {
            if (u01(blockRng) >= featureGen.chancePerGridCell)
            {
                continue;
            }

            if (!featureGen.possibleTopLayers.empty())
            {
                bool canPlace = false;
                for (const auto& possibleTopLayer : featureGen.possibleTopLayers)
                {
                    int layerIdx = (int)possibleTopLayer.material;
                    float layerStart = columnLayers[256 * layerIdx];
                    float layerEnd = columnLayers[256 * (layerIdx + 1)];

                    if (layerStart > height || layerEnd < height || min(layerEnd, height) - layerStart < possibleTopLayer.minThickness)
                    {
                        continue;
                    }

                    canPlace = true;
                    break;
                }

                if (!canPlace)
                {
                    continue;
                }
            }

            if (isFeaturePos(worldBlockPos2d, featureGen.gridCellSize, featureGen.gridCellPadding, (int)featureGen.feature * 518721))
            {
                this->featurePlacements.push_back({
                    featureGen.feature,
                    ivec3(worldBlockPos2d.x, groundHeight + 1, worldBlockPos2d.y),
                    featureGen.canReplaceBlocks
                });
                break;
            }
        }
    }
}

void Chunk::generateFeaturePlacements()
{
    for (int localZ = 0; localZ < 16; ++localZ)
    {
        for (int localX = 0; localX < 16; ++localX)
        {
            generateColumnFeaturePlacements(localX, localZ);
        }
    }
}

static const std::array<ivec2, 49> gatherFeaturePlacementsChunkOffsets = {
    ivec2(0, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, 0), ivec2(1, -1), ivec2(0, -1), ivec2(-1, -1),
    ivec2(-1, 0), ivec2(-1, 1), ivec2(2, 0), ivec2(2, 1), ivec2(2, 2), ivec2(1, 2), ivec2(0, 2),
    ivec2(-1, 2), ivec2(-2, 2), ivec2(-2, 1), ivec2(-2, 0), ivec2(-2, -1), ivec2(-2, -2),
    ivec2(-1, -2), ivec2(0, -2), ivec2(1, -2), ivec2(2, -2), ivec2(2, -1),
    ivec2(-3, -3), ivec2(-2, -3), ivec2(-1, -3), ivec2(0, -3), ivec2(1, -3), ivec2(2, -3), ivec2(3, -3),
    ivec2(3, -2), ivec2(3, -1), ivec2(3, 0), ivec2(3, 1), ivec2(3, 2), ivec2(3, 3),
    ivec2(2, 3), ivec2(1, 3), ivec2(0, 3), ivec2(-1, 3), ivec2(-2, 3), ivec2(-3, 3),
    ivec2(-3, 2), ivec2(-3, 1), ivec2(-3, 0), ivec2(-3, -1), ivec2(-3, -2)
};

void Chunk::otherChunkGatherFeaturePlacements(Chunk* chunkPtr, Chunk* const (&neighborChunks)[13][13], int centerX, int centerZ)
{
    chunkPtr->gatheredFeaturePlacements.clear();

    for (const auto& offset : gatherFeaturePlacementsChunkOffsets)
    {
        const auto& neighborPtr = neighborChunks[centerZ + offset.y][centerX + offset.x];

        for (const auto& neighborFeaturePlacement : neighborPtr->featurePlacements)
        {
            chunkPtr->gatheredFeaturePlacements.push_back(neighborFeaturePlacement);
        }
        
        for (const auto& neighborCaveFeaturePlacement : neighborPtr->caveFeaturePlacements)
        {
            chunkPtr->gatheredCaveFeaturePlacements.push_back(neighborCaveFeaturePlacement);
        }
    }
}

void Chunk::gatherFeaturePlacements()
{
    floodFillAndIterateNeighbors<13>(
        ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS,
        ChunkState::READY_TO_FILL,
        &Chunk::otherChunkGatherFeaturePlacements
    );
}

#pragma endregion

#pragma region chunk fill

__device__ void chunkFillPlaceBlock(
    Block* blockPtr,
    float* shared_biomeWeights,
    float* shared_layersAndHeight,
    CaveLayer* shared_caveLayers,
    int y,
    float height,
    ivec3 worldBlockPos,
    thrust::random::default_random_engine& rng)
{
    if (y == 0)
    {
        *blockPtr = Block::BEDROCK;
        return;
    }

    if (y > height && y > SEA_LEVEL)
    {
        *blockPtr = Block::AIR;
        return;
    }

    bool isOcean = false;
    for (int biomeIdx = 0; biomeIdx < numOceanBiomes; ++biomeIdx)
    {
        if (shared_biomeWeights[biomeIdx] > 0.f)
        {
            isOcean = true;
            break;
        }
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    Biome randBiome = getRandomBiome(shared_biomeWeights, u01(rng));
    bool isTopBlock = y >= height - 1.f;

#define doBlockPostProcess() biomeBlockPostProcess(blockPtr, randBiome, worldBlockPos, height, isTopBlock)
#ifdef DEBUG_CAVE_BIOME_OVERRIDE
#define postProcessCaveBiome DEBUG_CAVE_BIOME_OVERRIDE
#else
#define postProcessCaveBiome getCaveBiome(worldBlockPos, height, 190249401)
#endif
#define doCaveBlockPostProcess() caveBiomeBlockPostProcess(blockPtr, postProcessCaveBiome, worldBlockPos, caveBottomDepth, caveTopDepth)

    if (y > height && y <= SEA_LEVEL)
    {
        *blockPtr = Block::WATER;
        doBlockPostProcess();

        if (isOcean)
        {
            return;
        }
    }

    int caveBottomDepth = -384;
    int caveTopDepth = -384;
    int caveLayerIdx = 0;
    for ( ; caveLayerIdx < MAX_CAVE_LAYERS_PER_COLUMN; ++caveLayerIdx)
    {
        const auto& caveLayer = shared_caveLayers[caveLayerIdx];
        if (caveLayer.start == 384)
        {
            caveBottomDepth = -384; // all caves for this layer are under this block
            break;
        }

        caveBottomDepth = caveLayer.start - y;

        if (y <= caveLayer.start)
        {
            break;
        }

        // {{ y > caveLayer.start }}
        if (y <= caveLayer.end)
        {
            caveBottomDepth = caveLayer.start - y;
            caveTopDepth = y - (caveLayer.end + 1);
            *blockPtr = (y <= LAVA_LEVEL) ? Block::LAVA : Block::AIR;
            doCaveBlockPostProcess();
            return;
        }

        // {{ y > caveLayer.start /\ y > caveLayer.end }}
        caveTopDepth = y - (caveLayer.end + 1);
    }

    if (y > height)
    {
        return;
    }

    bool wasBlockPreProcessed = biomeBlockPreProcess(blockPtr, randBiome, worldBlockPos, height);
    if (wasBlockPreProcessed)
    {
        doBlockPostProcess();
        return;
    }

    int layerIdxStart;
    if (y >= shared_layersAndHeight[numForwardMaterials])
    {
        layerIdxStart = numForwardMaterials;
    }
    else
    {
        layerIdxStart = 0;
    }

#if DEBUG_USE_CONTRIBUTION_FILL_METHOD
    float maxContribution = 0.f;
    int maxLayerIdx = -1;
    #pragma unroll
    for (int layerIdx = layerIdxStart; layerIdx < numMaterials; ++layerIdx)
    {
        float layerContributionStart = max(shared_layersAndHeight[layerIdx], (float)y);
        float layerContributionEnd = min(shared_layersAndHeight[layerIdx + 1], (float)y + 1.f);
        float layerContribution = layerContributionEnd - layerContributionStart;

        if (layerContribution > maxContribution)
        {
            maxContribution = layerContribution;
            maxLayerIdx = layerIdx;
        }
    }

    if (height < y + 0.5f)
    {
        block = Block::AIR;
    }
    else
    {
        block = dev_materialInfos[maxLayerIdx].block;

        bool isTopBlock = height < y + 1.5f;
        if (isTopBlock)
        {
            if (block == Block::DIRT)
            {
                const Biome randBiome = getRandomBiome(shared_biomeWeights, u01(rng));
                block = dev_biomeBlocks[(int)randBiome].grassBlock;
            }
        }
    }
#else
    int thisLayerIdx = -1;
    #pragma unroll
    for (int layerIdx = layerIdxStart; layerIdx < numMaterials; ++layerIdx)
    {
        float layerStart = shared_layersAndHeight[layerIdx];
        float layerEnd = shared_layersAndHeight[layerIdx + 1];

        if (layerStart <= y && y < layerEnd)
        {
            thisLayerIdx = layerIdx;
            break;
        }
    }

    *blockPtr = dev_materialInfos[thisLayerIdx].block;

    if (isTopBlock)
    {
        if (*blockPtr == Block::DIRT)
        {
            *blockPtr = dev_biomeBlocks[(int)randBiome].grassBlock;
        }
    }
#endif

    doBlockPostProcess();
    doCaveBlockPostProcess();

#undef doBlockPostProcess
#undef doCaveBlockPostProcess
#undef postProcessCaveBiome
}

__global__ void kernFill(
    Block* blocks,
    float* heightfield,
    float* biomeWeights,
    float* layers,
    CaveLayer* caveLayers,
    FeaturePlacement* featurePlacements,
    ivec2 allFeaturesHeightBounds,
    CaveFeaturePlacement* caveFeaturePlacements,
    ivec2 allCaveFeaturesHeightBounds,
    ivec3 chunkWorldBlockPos)
{
    __shared__ float shared_biomeWeights[numBiomes];
    __shared__ float shared_layersAndHeight[numMaterials + 1];
    __shared__ CaveLayer shared_caveLayers[MAX_CAVE_LAYERS_PER_COLUMN];

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posTo3dIndex(x, y, z);
    const int idx2d = posTo2dIndex(x, z);

    int loadIdx = threadIdx.y;
    float* floatLoadLocation = nullptr;
    float* floatStoreLocation = nullptr;
    if (loadIdx < numBiomes)
    {
        floatLoadLocation = biomeWeights + (idx2d) + (256 * loadIdx);
        floatStoreLocation = shared_biomeWeights + loadIdx;
    }
    else if ((loadIdx -= numBiomes) <= numMaterials)
    {
        floatLoadLocation = loadIdx == numMaterials ? (heightfield + idx2d) : (layers + (idx2d) + (256 * loadIdx));
        floatStoreLocation = shared_layersAndHeight + loadIdx;
    }
    else if ((loadIdx -= numMaterials + 1) < MAX_CAVE_LAYERS_PER_COLUMN)
    {
        shared_caveLayers[loadIdx] = caveLayers[(MAX_CAVE_LAYERS_PER_COLUMN * idx2d) + loadIdx];
    }

    if (floatStoreLocation != nullptr)
    {
        *floatStoreLocation = *floatLoadLocation;
    }

    __syncthreads();

    const float height = shared_layersAndHeight[numMaterials];

    const ivec3 worldBlockPos = chunkWorldBlockPos + ivec3(x, y, z);
    auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z);

    Block block;
    chunkFillPlaceBlock(&block, shared_biomeWeights, shared_layersAndHeight, shared_caveLayers, y, height, worldBlockPos, rng);

    bool isInFeatureBounds = y >= allFeaturesHeightBounds[0] && y <= allFeaturesHeightBounds[1];
    bool isInCaveFeatureBounds = y >= allCaveFeaturesHeightBounds[0] && y <= allCaveFeaturesHeightBounds[1];

    Block featureBlock;
    bool placedFeature = false;
    if (isInFeatureBounds)
    {
        for (int featureIdx = 0; featureIdx < MAX_GATHERED_FEATURES_PER_CHUNK; ++featureIdx)
        {
            const auto& featurePlacement = featurePlacements[featureIdx];

            if (featurePlacement.feature == Feature::NONE)
            {
                break;
            }

            if (block != Block::AIR && !featurePlacement.canReplaceBlocks)
            {
                continue;
            }

            ivec2 featureHeightBounds = dev_featureHeightBounds[(int)featurePlacement.feature] + ivec2(featurePlacement.pos.y);
            if (y < featureHeightBounds[0] || y > featureHeightBounds[1])
            {
                continue;
            }

            if (placeFeature(featurePlacement, worldBlockPos, &featureBlock))
            {
                placedFeature = true;
                break;
            }
        }
    }

    if (isInCaveFeatureBounds && !placedFeature)
    {
        for (int caveFeatureIdx = 0; caveFeatureIdx < MAX_GATHERED_CAVE_FEATURES_PER_CHUNK; ++caveFeatureIdx)
        {
            const auto& caveFeaturePlacement = caveFeaturePlacements[caveFeatureIdx];

            if (caveFeaturePlacement.feature == CaveFeature::NONE)
            {
                break;
            }

            if (block != Block::AIR && !caveFeaturePlacement.canReplaceBlocks)
            {
                continue;
            }

            const int featureY = caveFeaturePlacement.pos.y;
            ivec2 caveFeatureHeightBounds = ivec2(featureY, featureY + caveFeaturePlacement.layerHeight) + dev_caveFeatureHeightBounds[(int)caveFeaturePlacement.feature];
            if (y < caveFeatureHeightBounds[0] || y > caveFeatureHeightBounds[1])
            {
                continue;
            }

            if (placeCaveFeature(caveFeaturePlacement, worldBlockPos, &featureBlock))
            {
                placedFeature = true;
                break;
            }
        }
    }

    if (placedFeature)
    {
        block = featureBlock;
    }

    blocks[idx] = block;
}

void heightBoundsMinMax(ivec2& in, const ivec2& v)
{
    in[0] = min(in[0], v[0]);
    in[1] = max(in[1], v[1]);
}

void Chunk::fill(
    std::vector<Chunk*>& chunks,
    float* host_heightfields,
    float* dev_heightfields,
    float* host_biomeWeights,
    float* dev_biomeWeights,
    float* host_layers,
    float* dev_layers,
    CaveLayer* host_caveLayers,
    CaveLayer* dev_caveLayers,
    FeaturePlacement* dev_featurePlacements,
    CaveFeaturePlacement* dev_caveFeaturePlacements,
    Block* host_blocks,
    Block* dev_blocks,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(host_heightfields + (i * 256), chunkPtr->heightfield.data(), 256 * sizeof(float));
        std::memcpy(host_biomeWeights + (i * devBiomeWeightsSize), chunkPtr->biomeWeights.data(), devBiomeWeightsSize * sizeof(float));
        std::memcpy(host_layers + (i * devLayersSize), chunkPtr->layers.data(), devLayersSize * sizeof(float));
        std::memcpy(host_caveLayers + (i * devCaveLayersSize), chunkPtr->caveLayers.data(), devCaveLayersSize * sizeof(CaveLayer));
    }

    cudaMemcpyAsync(dev_heightfields, host_heightfields, numChunks * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_biomeWeights, host_biomeWeights, numChunks * devBiomeWeightsSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_layers, host_layers, numChunks * devLayersSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_caveLayers, host_caveLayers, numChunks * devCaveLayersSize * sizeof(CaveLayer), cudaMemcpyHostToDevice, stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        ivec2 allFeaturesHeightBounds = ivec2(384, -1);
        for (const auto& featurePlacement : chunkPtr->gatheredFeaturePlacements)
        {
            const auto& featureHeightBounds = host_featureHeightBounds[(int)featurePlacement.feature];
            const ivec2 thisFeatureHeightBounds = ivec2(featurePlacement.pos.y) + featureHeightBounds;
            heightBoundsMinMax(allFeaturesHeightBounds, thisFeatureHeightBounds);
        }

        ivec2 allCaveFeaturesHeightBounds = ivec2(384, -1);
        for (const auto& caveFeaturePlacement : chunkPtr->gatheredCaveFeaturePlacements)
        {
            const auto& caveFeatureHeightBounds = host_caveFeatureHeightBounds[(int)caveFeaturePlacement.feature];
            const int featureY = caveFeaturePlacement.pos.y;
            const ivec2 thisCaveFeatureHeightBounds = ivec2(featureY, featureY + caveFeaturePlacement.layerHeight) + caveFeatureHeightBounds;
            heightBoundsMinMax(allCaveFeaturesHeightBounds, thisCaveFeatureHeightBounds);
        }

        // TODO: extract repeated code to a function
        int numFeaturePlacements = min((int)chunkPtr->gatheredFeaturePlacements.size(), MAX_GATHERED_FEATURES_PER_CHUNK);
        if (numFeaturePlacements < MAX_GATHERED_FEATURES_PER_CHUNK)
        {
            chunkPtr->gatheredFeaturePlacements.push_back({ Feature::NONE });
            ++numFeaturePlacements;
        }
        cudaMemcpyAsync(
            dev_featurePlacements + (i * MAX_GATHERED_FEATURES_PER_CHUNK),
            chunkPtr->gatheredFeaturePlacements.data(),
            numFeaturePlacements * sizeof(FeaturePlacement),
            cudaMemcpyHostToDevice,
            stream
        );
        chunkPtr->gatheredFeaturePlacements.clear();

        int numCaveFeaturePlacements = min((int)chunkPtr->gatheredCaveFeaturePlacements.size(), MAX_GATHERED_CAVE_FEATURES_PER_CHUNK);
        if (numCaveFeaturePlacements < MAX_GATHERED_CAVE_FEATURES_PER_CHUNK)
        {
            chunkPtr->gatheredCaveFeaturePlacements.push_back({ CaveFeature::NONE });
            ++numCaveFeaturePlacements;
        }
        cudaMemcpyAsync(
            dev_caveFeaturePlacements + (i * MAX_GATHERED_CAVE_FEATURES_PER_CHUNK),
            chunkPtr->gatheredCaveFeaturePlacements.data(),
            numCaveFeaturePlacements * sizeof(CaveFeaturePlacement),
            cudaMemcpyHostToDevice,
            stream
        );
        chunkPtr->gatheredCaveFeaturePlacements.clear();

        const dim3 blockSize3d(1, 128, 1);
        const dim3 blocksPerGrid3d(16, 3, 16);
        kernFill<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
            dev_blocks + (i * devBlocksSize),
            dev_heightfields + (i * 256),
            dev_biomeWeights + (i * devBiomeWeightsSize),
            dev_layers + (i * devLayersSize),
            dev_caveLayers + (i * devCaveLayersSize),
            dev_featurePlacements + (i * MAX_GATHERED_FEATURES_PER_CHUNK),
            allFeaturesHeightBounds,
            dev_caveFeaturePlacements + (i * MAX_GATHERED_CAVE_FEATURES_PER_CHUNK),
            allCaveFeaturesHeightBounds,
            chunkPtr->worldBlockPos
        );
    }

    cudaMemcpyAsync(host_blocks, dev_blocks, numChunks * devBlocksSize * sizeof(Block), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->blocks.data(), host_blocks + (i * devBlocksSize), devBlocksSize * sizeof(Block));
        chunkPtr->placeDecorators();
    }

    CudaUtils::checkCUDAError("Chunk::fill() failed");
}

void Chunk::tryPlaceSingleDecorator(
    ivec3 pos, 
    const DecoratorGen& gen)
{
    const int decoratorIdx = posTo3dIndex(pos);
    Block& currentBlock = this->blocks[decoratorIdx];
    if (!gen.possibleReplaceBlocks.empty() 
        && gen.possibleReplaceBlocks.find(currentBlock) == gen.possibleReplaceBlocks.end())
    {
        return;
    }

    int underBlockOffset = gen.generatesFromCeiling ? 1 : -1;
    if (!isInRange(pos.y + underBlockOffset, 0, 383))
    {
        return;
    }

    const Block underBlock = blocks[decoratorIdx + underBlockOffset];
    if ((int)underBlock < numNonSolidBlocks
        || (!gen.possibleUnderBlocks.empty() && gen.possibleUnderBlocks.find(underBlock) == gen.possibleUnderBlocks.end()))
    {
        return;
    }

    if (gen.secondDecoratorBlock != Block::AIR)
    {
        int overBlockOffset = -underBlockOffset;
        if (!isInRange(pos.y + overBlockOffset, 0, 383))
        {
            return;
        }

        Block& overBlock = this->blocks[decoratorIdx + overBlockOffset];
        if (!gen.possibleReplaceBlocks.empty() && gen.possibleReplaceBlocks.find(overBlock) == gen.possibleReplaceBlocks.end())
        {
            return;
        }

        overBlock = gen.secondDecoratorBlock;
    }

    currentBlock = gen.decoratorBlock;
}

void Chunk::placeDecorators()
{
    auto rng = makeSeededRandomEngine(this->worldBlockPos.x, this->worldBlockPos.y, this->worldBlockPos.z, 7589341);
    thrust::uniform_real_distribution<float> u01(0, 1);

    for (int z = 0; z < 16; ++z)
    {
        for (int x = 0; x < 16; ++x)
        {
            const int idx2d = posTo2dIndex(x, z);

            const float* columnBiomeWeights = biomeWeights.data() + idx2d;
            Biome biome = getRandomBiome<256>(columnBiomeWeights, u01(rng));

            float rand = u01(rng);
            const auto& biomeDecoratorGens = host_biomeDecoratorGens[(int)biome];
            for (int genIdx = 0; genIdx < biomeDecoratorGens.size(); ++genIdx)
            {
                const auto& gen = biomeDecoratorGens[genIdx];

                if ((rand -= gen.chance) < 0.f)
                {
                    tryPlaceSingleDecorator(ivec3(x, ((int)this->heightfield[idx2d]) + 1, z), gen);
                    break;
                }
            }

            const CaveLayer* columnCaveLayers = this->caveLayers.data() + (MAX_CAVE_LAYERS_PER_COLUMN * idx2d);
            for (int caveLayerIdx = 0; caveLayerIdx < MAX_CAVE_LAYERS_PER_COLUMN; ++caveLayerIdx)
            {
                const auto& caveLayer = columnCaveLayers[caveLayerIdx];

                if (caveLayer.start == 384)
                {
                    break;
                }

                float bottomRand = u01(rng);
                float topRand = u01(rng);
                bool placedBottom = false;
                bool placedTop = false;
                const auto& caveBiomeDecoratorGens = host_caveBiomeDecoratorGens[(int)caveLayer.bottomBiome];
                for (int genIdx = 0; genIdx < caveBiomeDecoratorGens.size(); ++genIdx)
                {
                    const auto& gen = caveBiomeDecoratorGens[genIdx];
                    if (gen.generatesFromCeiling)
                    {
                        if (!placedTop && (topRand -= gen.chance) < 0.f)
                        {
                            tryPlaceSingleDecorator(ivec3(x, caveLayer.end, z), gen);
                        }
                    }
                    else
                    {
                        if (!placedBottom && (bottomRand -= gen.chance) < 0.f)
                        {
                            tryPlaceSingleDecorator(ivec3(x, caveLayer.start + 1, z), gen);
                        }
                    }

                    if (placedTop && placedBottom)
                    {
                        break;
                    }
                }
            }
        }
    }
}

#pragma endregion

#pragma region VBOs

static const float xShapedPosOffset = 0.5f * sinf(glm::radians(45.f));
static const std::array<vec3, 8> xShapedVertPositions = {
    vec3(xShapedPosOffset, 0.f, xShapedPosOffset),
    vec3(-xShapedPosOffset, 0.f, -xShapedPosOffset),
    vec3(-xShapedPosOffset, 1.f, -xShapedPosOffset),
    vec3(xShapedPosOffset, 1.f, xShapedPosOffset),

    vec3(-xShapedPosOffset, 0.f, xShapedPosOffset),
    vec3(xShapedPosOffset, 0.f, -xShapedPosOffset),
    vec3(xShapedPosOffset, 1.f, -xShapedPosOffset),
    vec3(-xShapedPosOffset, 1.f, xShapedPosOffset)
};
static const vec3 xShapedFaceNormal1 = normalize(vec3(1, 0, -1));
static const vec3 xShapedFaceNormal2 = normalize(vec3(1, 0, 1));

static const std::array<ivec3, 24> directionVertPositions = {
    ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 1),
    ivec3(1, 0, 1), ivec3(1, 0, 0), ivec3(1, 1, 0), ivec3(1, 1, 1),
    ivec3(1, 0, 0), ivec3(0, 0, 0), ivec3(0, 1, 0), ivec3(1, 1, 0),
    ivec3(0, 0, 0), ivec3(0, 0, 1), ivec3(0, 1, 1), ivec3(0, 1, 0),
    ivec3(0, 1, 1), ivec3(1, 1, 1), ivec3(1, 1, 0), ivec3(0, 1, 0),
    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(1, 0, 1), ivec3(0, 0, 1)
};

static const std::array<ivec2, 4> uvOffsets = {
    ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 1)
};

void Chunk::createVBOs()
{
    idx.clear();
    verts.clear();

    idxCount = 0;

    for (int z = 0; z < 16; ++z)
    {
        for (int x = 0; x < 16; ++x)
        {
            for (int y = 0; y < 384; ++y)
            {
                ivec3 thisPos = ivec3(x, y, z);
                Block thisBlock = blocks[posTo3dIndex(thisPos)];

                Mats mat;
                switch (thisBlock)
                {
                case Block::AIR:
                    continue;
                case Block::WATER:
                    mat = Mats::M_WATER;
                    break;
                case Block::CYAN_CRYSTAL:
                case Block::GREEN_CRYSTAL:
                case Block::MAGENTA_CRYSTAL:
                    mat = Mats::M_CRYSTAL;
                    break;
                case Block::MARBLE:
                case Block::QUARTZ:
                case Block::ICE:
                case Block::PACKED_ICE:
                case Block::BLUE_ICE:
                    mat = Mats::M_SMOOTH_MICRO;
                    break;
                case Block::SNOW:
                case Block::SNOWY_GRASS_BLOCK: // really should only apply to the snowy part but whatever
                    mat = Mats::M_MICRO;
                    break;
                case Block::SAND:
                case Block::GRAVEL:
                    mat = Mats::M_ROUGH_MICRO;
                    break;
                default:
                    mat = Mats::M_DIFFUSE;
                    break;
                }

                BlockData thisBlockData = BlockUtils::getBlockData(thisBlock);
                const auto thisTrans = thisBlockData.transparency;

                if (thisTrans == TransparencyType::T_X_SHAPED)
                {
                    vec3 basePos = vec3(x + 0.5f, y, z + 0.5f);

                    vec2 worldBlockXZ = vec2(this->worldBlockPos.x + x, this->worldBlockPos.z + z);
                    vec2 randomOffset = 0.4f * (rand2From2(worldBlockXZ) - 0.5f);
                    basePos.x += randomOffset.x;
                    basePos.z += randomOffset.y;

                    int idx1 = verts.size();

                    for (int i = 0; i < 8; i++)
                    {
                        auto posOffset = xShapedVertPositions[i];

                        verts.emplace_back();
                        Vertex& vert = verts.back();

                        vert.pos = basePos + posOffset;
                        vert.nor = i < 4 ? xShapedFaceNormal1 : xShapedFaceNormal2;
                        vert.uv = vec2(thisBlockData.uvs.side.uv + uvOffsets[i % 4]) * 0.0625f;
                        vert.m = mat;
                    }

                    idx.push_back(idx1);
                    idx.push_back(idx1 + 1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1 + 3);

                    idx.push_back(idx1 + 4);
                    idx.push_back(idx1 + 5);
                    idx.push_back(idx1 + 6);
                    idx.push_back(idx1 + 4);
                    idx.push_back(idx1 + 6);
                    idx.push_back(idx1 + 7);

                    continue;
                }

                for (int dirIdx = 0; dirIdx < 6; ++dirIdx)
                {
                    const auto& direction = DirectionEnums::dirVecs[dirIdx];
                    ivec3 neighborPos = thisPos + direction;
                    Chunk* neighborPosChunk = this;
                    Block neighborBlock;

                    if (neighborPos.y >= 0 && neighborPos.y < 384)
                    {
                        if (neighborPos.x < 0)
                        {
                            neighborPosChunk = neighbors[3];
                            neighborPos.x += 16;
                        }
                        else if (neighborPos.x >= 16)
                        {
                            neighborPosChunk = neighbors[1];
                            neighborPos.x -= 16;
                        }
                        else if (neighborPos.z < 0)
                        {
                            neighborPosChunk = neighbors[2];
                            neighborPos.z += 16;
                        }
                        else if (neighborPos.z >= 16)
                        {
                            neighborPosChunk = neighbors[0];
                            neighborPos.z -= 16;
                        }

                        if (neighborPosChunk == nullptr)
                        {
                            continue;
                        }

                        neighborBlock = neighborPosChunk->blocks[posTo3dIndex(neighborPos)];

                        const auto neighborTrans = BlockUtils::getBlockData(neighborBlock).transparency;

                        // OPAQUE displays if neighbor is not OPAQUE
                        // SEMI_TRANSPARENT if neighbor is not OPAQUE
                        // TRANSPARENT (except AIR) displays if neighbor is AIR or SEMI_TRANSPARENT (may need to revise this if two different transparent blocks are adjacent)
                        // X_SHAPED displays no matter what (handled above)
                        bool shouldDisplay;
                        switch (thisTrans)
                        {
                        case TransparencyType::T_OPAQUE:
                        case TransparencyType::T_SEMI_TRANSPARENT:
                            shouldDisplay = neighborTrans != TransparencyType::T_OPAQUE;
                            break;
                        case TransparencyType::T_TRANSPARENT:
                            shouldDisplay = neighborBlock == Block::AIR || neighborTrans == TransparencyType::T_SEMI_TRANSPARENT;
                            break;
                        }

                        if (!shouldDisplay)
                        {
                            continue;
                        }
                    }

                    int idx1 = verts.size();

                    const auto& thisUvs = thisBlockData.uvs;
                    SideUv sideUv;
                    switch (direction.y)
                    {
                    case 1:
                        sideUv = thisUvs.top;
                        break;
                    case -1:
                        sideUv = thisUvs.bottom;
                        break;
                    case 0:
                        sideUv = thisUvs.side;
                        break;
                    }

                    int uvStartIdx = 0;
                    int uvFlipIdx = -1;
                    if (sideUv.randRot || sideUv.randFlip)
                    {
                        ivec3 worldPos = thisPos + this->worldBlockPos;
                        auto rng = makeSeededRandomEngine(worldPos.x, worldPos.y, worldPos.z, dirIdx);
                        thrust::uniform_real_distribution<float> u04(0, 4);
                        if (sideUv.randRot)
                        {
                            uvStartIdx = (int)u04(rng);
                        }
                        if (sideUv.randFlip)
                        {
                            uvFlipIdx = (int)u04(rng);
                        }
                    }

                    for (int j = 0; j < 4; ++j)
                    {
                        verts.emplace_back();
                        Vertex& vert = verts.back();

                        vert.pos = vec3(thisPos + directionVertPositions[dirIdx * 4 + j]);
                        vert.nor = direction;

                        ivec2 uvOffset = uvOffsets[(uvStartIdx + j) % 4];
                        if (uvFlipIdx != -1)
                        {
                            if (uvFlipIdx & 1)
                            {
                                uvOffset.x = 1 - uvOffset.x;
                            }
                            if (uvFlipIdx & 2)
                            {
                                uvOffset.y = 1 - uvOffset.y;
                            }
                        }
                        vert.uv = vec2(sideUv.uv + uvOffset) * 0.0625f;
                        vert.m = mat;
                    }

                    idx.push_back(idx1);
                    idx.push_back(idx1 + 1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1);
                    idx.push_back(idx1 + 2);
                    idx.push_back(idx1 + 3);
                }
            }
        }
    }
}

void Chunk::bufferVBOs()
{
    idxCount = idx.size();

    generateIdx();
    bindIdx();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    generateVerts();
    bindVerts();
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

    idx.clear();
    verts.clear();
}

#pragma endregion
