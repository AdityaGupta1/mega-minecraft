#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "biomeFuncs.hpp"
#include "featurePlacement.hpp"
#include "util/rng.hpp"
#include "defines.hpp"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define DEBUG_SKIP_EROSION 0
#define DEBUG_USE_CONTRIBUTION_FILL_METHOD 0

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
    const int radius = diameter / 2;

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
    int start = diameter / 4; // assuming diameter = (4k + 1) for some k
    int end = diameter - start;

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
                    }
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
    ivec2* dev_chunkWorldBlockPositions,
    float* dev_heightfields,
    float* dev_biomeWeights,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    ivec2* host_chunkWorldBlockPositions = new ivec2[numChunks];
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

    float* host_heightfields = new float[numChunks * 256];
    float* host_biomeWeights = new float[numChunks * devBiomeWeightsSize];

    cudaMemcpyAsync(host_heightfields, dev_heightfields, numChunks * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_biomeWeights, dev_biomeWeights, numChunks * devBiomeWeightsSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
  
    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->heightfield.data(), host_heightfields + (256 * i), 256 * sizeof(float));
        std::memcpy(chunkPtr->biomeWeights.data(), host_biomeWeights + (devBiomeWeightsSize * i), devBiomeWeightsSize * sizeof(float));
    }

    delete[] host_chunkWorldBlockPositions;
    delete[] host_heightfields;
    delete[] host_biomeWeights;

    CudaUtils::checkCUDAError("Chunk::generateHeightfield() failed");
}

void calculateEdgeIndices(int offset, int& in, int& out)
{
    in = (offset == -1) ? 15 : 0;
    out = (offset == -1) ? 0 : 17;
}

void Chunk::otherChunkGatherHeightfield(Chunk* chunkPtr, Chunk* const (&neighborChunks)[5][5], int centerX, int centerZ)
{
    chunkPtr->gatheredHeightfield.reserve(18 * 18);

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
        for (int x = 0; x < 16; ++x)
        {
            chunkPtr->gatheredHeightfield[posTo2dIndex<18>(x + 1, z + 1)] = chunkPtr->heightfield[posTo2dIndex(x, z)];
        }
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
    float* dev_heightfields,
    float* dev_biomeWeights,
    ivec2* dev_chunkWorldBlockPositions,
    float* dev_layers,
    cudaStream_t stream)
{
    const int numChunks = chunks.size();

    float* host_heightfields = new float[numChunks * devHeightfieldSize];
    float* host_biomeWeights = new float[numChunks * devBiomeWeightsSize];
    ivec2* host_chunkWorldBlockPositions = new ivec2[numChunks];
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

    float* host_layers = new float[numChunks * devLayersSize];

    cudaMemcpyAsync(host_layers, dev_layers, numChunks * devLayersSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < numChunks; ++i)
    {
        Chunk* chunkPtr = chunks[i];

        std::memcpy(chunkPtr->layers.data(), host_layers + (i * devLayersSize), devLayersSize * sizeof(float));
    }

    delete[] host_heightfields;
    delete[] host_biomeWeights;
    delete[] host_chunkWorldBlockPositions;
    delete[] host_layers;

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

void Chunk::erodeZone(Zone* zonePtr, float* dev_gatheredLayers, float* dev_accumulatedHeights, cudaStream_t stream)
{
#if !DEBUG_SKIP_EROSION
    float* gatheredLayers = new float[gatheredLayersBaseSize];
    copyLayers(zonePtr, gatheredLayers, true);
    zonePtr->gatheredChunks.clear();

    int gatheredLayersSizeBytes = gatheredLayersBaseSize * sizeof(float);
    cudaMemcpyAsync(dev_gatheredLayers, gatheredLayers, gatheredLayersSizeBytes, cudaMemcpyHostToDevice, stream);

    float flagDidChange;
    float* dev_flagDidChange = dev_gatheredLayers + gatheredLayersBaseSize;

    const dim3 blockSize2d(32, 32);
    constexpr int blocksPerGrid = (ZONE_SIZE * 2 * 16) / 32; // = ZONE_SIZE but writing it out for clarity
    const dim3 blocksPerGrid2d(blocksPerGrid, blocksPerGrid);

    thrust::device_ptr<float> dev_ptr(dev_accumulatedHeights);
    thrust::fill(dev_ptr, dev_ptr + EROSION_GRID_NUM_COLS, 0.f);

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

    cudaMemcpyAsync(gatheredLayers, dev_gatheredLayers, gatheredLayersSizeBytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // all data needs to be copied back to gatheredLayers before calling copyLayers()
                                   // explicit synchronization here may not be necessary (seems to work without it) but it gives peace of mind

    copyLayers(zonePtr, gatheredLayers, false);
    delete[] gatheredLayers;
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

#pragma region feature placements

// TODO: maybe revisit and try to optimize this by calculating thicknesses in a separate pass with better memory access patterns
void Chunk::generateFeaturePlacements()
{
    for (int localZ = 0; localZ < 16; ++localZ)
    {
        for (int localX = 0; localX < 16; ++localX)
        {
            const int idx2d = posTo2dIndex(localX, localZ);

            const auto& columnBiomeWeights = biomeWeights.data() + idx2d;

            const ivec3 worldBlockPos = this->worldBlockPos + ivec3(localX, heightfield[idx2d], localZ);
            auto blockRng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z, 7); // arbitrary w so this rng is different than heightfield rng
            thrust::uniform_real_distribution<float> u01(0, 1);

            Biome biome = getRandomBiome<256>(columnBiomeWeights, u01(blockRng));
            const auto& featureGens = BiomeUtils::getBiomeFeatureGens(biome);

            const float* columnLayers = this->layers.data() + idx2d;
            Material topLayerMaterial;
            float topLayerThickness;

            const float lastLayerThickness = this->heightfield[idx2d] - columnLayers[256 * (numMaterials - 1)];
            if (lastLayerThickness > 0)
            {
                topLayerMaterial = (Material)(numMaterials - 1);
                topLayerThickness = lastLayerThickness;
            }
            else
            {
                for (int layerIdx = numMaterials - 2; layerIdx >= 0; --layerIdx)
                {
                    const float thickness = columnLayers[256 * (layerIdx + 1)] - columnLayers[256 * (layerIdx)];
                    if (thickness > 0)
                    {
                        topLayerMaterial = (Material)layerIdx;
                        topLayerThickness = thickness;
                        break;
                    }
                }
            }

            const ivec2 localPos2d = ivec2(localX, localZ);
            const ivec2 worldPos2d = localPos2d + ivec2(this->worldBlockPos.x, this->worldBlockPos.z);

            Feature feature = Feature::NONE;
            //float rand = u01(rng);
            for (int i = 0; i < featureGens.size(); ++i)
            {
                const auto& featureGen = featureGens[i];

                bool canPlace = false;
                for (const auto& possibleTopLayer : featureGen.possibleTopLayers)
                {
                    if (topLayerMaterial == possibleTopLayer.material && topLayerThickness >= possibleTopLayer.minThickness)
                    {
                        canPlace = true;
                        break;
                    }
                }

                if (!canPlace)
                {
                    continue;
                }

                const ivec2 gridCornerWorldPos = ivec2(floor(vec2(worldPos2d) / (float)featureGen.gridCellSize) * (float)featureGen.gridCellSize);
                const int gridCellInternalSideLength = featureGen.gridCellSize - (2 * featureGen.gridCellPadding);
                vec2 randPos = rand2From3(vec3(gridCornerWorldPos, (int)featureGen.feature * 59321));
                const ivec2 gridPlaceWorldPos = gridCornerWorldPos + ivec2(featureGen.gridCellPadding)
                    + ivec2(floor(randPos * (float)gridCellInternalSideLength));
                const ivec2 gridPlaceLocalPos = gridPlaceWorldPos - ivec2(this->worldBlockPos.x, this->worldBlockPos.z);

                if (localPos2d == gridPlaceLocalPos && u01(blockRng) < featureGen.chancePerGridCell)
                {
                    feature = featureGen.feature;
                    break;
                }
            }

            if (feature != Feature::NONE)
            {
                this->featurePlacements.push_back({ feature, worldBlockPos });
            }
        }
    }

    // this probably won't include decorators (single block/column things) since those can be done on the CPU at the end of Chunk::fill()
}

void Chunk::otherChunkGatherFeaturePlacements(Chunk* chunkPtr, Chunk* const (&neighborChunks)[9][9], int centerX, int centerZ)
{
    chunkPtr->gatheredFeaturePlacements.clear();

    for (int offsetZ = -2; offsetZ <= 2; ++offsetZ)
    {
        for (int offsetX = -2; offsetX <= 2; ++offsetX)
        {
            const auto& neighborPtr = neighborChunks[centerZ + offsetZ][centerX + offsetX];

            for (const auto& neighborFeaturePlacement : neighborPtr->featurePlacements)
            {
                chunkPtr->gatheredFeaturePlacements.push_back(neighborFeaturePlacement);
            }
        }
    }
}

void Chunk::gatherFeaturePlacements()
{
    floodFillAndIterateNeighbors<9>(
        ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS,
        ChunkState::READY_TO_FILL,
        &Chunk::otherChunkGatherFeaturePlacements
    );
}

#pragma endregion

#pragma region chunk fill

__global__ void kernFill(
    Block* blocks,
    float* heightfield,
    float* biomeWeights,
    float* layers,
    FeaturePlacement* dev_featurePlacements,
    int numFeaturePlacements,
    ivec2 featureHeightBounds,
    ivec3 chunkWorldBlockPos)
{
    __shared__ float shared_layersAndHeight[numMaterials + 1];
    __shared__ float shared_biomeWeights[numBiomes];

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posTo3dIndex(x, y, z);
    const int idx2d = posTo2dIndex(x, z);

    float* loadLocation = nullptr;
    float* storeLocation = nullptr;
    if (threadIdx.y <= numMaterials)
    {
        loadLocation = threadIdx.y == numMaterials ? (heightfield + idx2d) : (layers + (idx2d) + (256 * threadIdx.y));
        storeLocation = shared_layersAndHeight + threadIdx.y;
    }
    else
    {
        const int biomeIdx = threadIdx.y - numMaterials - 1;
        if (biomeIdx < numBiomes)
        {
            loadLocation = biomeWeights + (idx2d) + (256 * biomeIdx);
            storeLocation = shared_biomeWeights + biomeIdx;
        }
    }

    if (storeLocation != nullptr)
    {
        *storeLocation = *loadLocation;
    }

    __syncthreads();

    const float height = shared_layersAndHeight[numMaterials];

    const ivec3 worldBlockPos = chunkWorldBlockPos + ivec3(x, y, z);
    auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Block block = Block::AIR;
    if (y < height)
    {
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

        block = dev_materialInfos[thisLayerIdx].block;

        bool isTopBlock = y >= height - 1.f;
        if (isTopBlock)
        {
            if (block == Block::DIRT)
            {
                const Biome randBiome = getRandomBiome(shared_biomeWeights, u01(rng));
                block = dev_biomeBlocks[(int)randBiome].grassBlock;
            }
        }
#endif
    }

    if (y < featureHeightBounds[0] || y > featureHeightBounds[1])
    {
        blocks[idx] = block;
        return;
    }

    Block featureBlock;
    bool placedFeature = false;
    for (int featureIdx = 0; featureIdx < numFeaturePlacements; ++featureIdx)
    {
        if (placeFeature(dev_featurePlacements[featureIdx], worldBlockPos, &featureBlock))
        {
            placedFeature = true;
            break;
        }
    }

    if (placedFeature)
    {
        block = featureBlock;
    }

    blocks[idx] = block;
}

void Chunk::fill(
    Block* dev_blocks, 
    float* dev_heightfield,
    float* dev_biomeWeights,
    float* dev_layers,
    FeaturePlacement* dev_featurePlacements, 
    cudaStream_t stream)
{
    ivec2 allFeaturesHeightBounds = ivec2(384, -1);
    for (const auto& featurePlacement : this->gatheredFeaturePlacements)
    {
        const auto& featureHeightBounds = BiomeUtils::getFeatureHeightBounds(featurePlacement.feature);
        const ivec2 thisFeatureHeightBounds = ivec2(featurePlacement.pos.y) + featureHeightBounds;
        allFeaturesHeightBounds[0] = min(allFeaturesHeightBounds[0], thisFeatureHeightBounds[0]);
        allFeaturesHeightBounds[1] = max(allFeaturesHeightBounds[1], thisFeatureHeightBounds[1]);
    }

    int numFeaturePlacements = min((int)this->gatheredFeaturePlacements.size(), MAX_GATHERED_FEATURES_PER_CHUNK);
    cudaMemcpyAsync(dev_featurePlacements, this->gatheredFeaturePlacements.data(), numFeaturePlacements * sizeof(FeaturePlacement), cudaMemcpyHostToDevice, stream);
    this->gatheredFeaturePlacements.clear();

    cudaMemcpyAsync(dev_heightfield, this->heightfield.data(), 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_layers, this->layers.data(), 256 * numMaterials * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_biomeWeights, this->biomeWeights.data(), 256 * numBiomes * sizeof(float), cudaMemcpyHostToDevice, stream);

    const dim3 blockSize3d(1, 128, 1);
    const dim3 blocksPerGrid3d(16, 3, 16);
    kernFill<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
        dev_blocks, 
        dev_heightfield,
        dev_biomeWeights,
        dev_layers,
        dev_featurePlacements,
        numFeaturePlacements,
        allFeaturesHeightBounds,
        this->worldBlockPos
    );
    
    cudaMemcpyAsync(this->blocks.data(), dev_blocks, 98304 * sizeof(Block), cudaMemcpyDeviceToHost, stream);

    CudaUtils::checkCUDAError("Chunk::fill() failed");
}

#pragma endregion

#pragma region VBOs

static const std::array<ivec3, 24> directionVertPositions = {
    ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 1),
    ivec3(1, 0, 1), ivec3(1, 0, 0), ivec3(1, 1, 0), ivec3(1, 1, 1),
    ivec3(1, 0, 0), ivec3(0, 0, 0), ivec3(0, 1, 0), ivec3(1, 1, 0),
    ivec3(0, 0, 0), ivec3(0, 0, 1), ivec3(0, 1, 1), ivec3(0, 1, 0),
    ivec3(0, 1, 1), ivec3(1, 1, 1), ivec3(1, 1, 0), ivec3(0, 1, 0),
    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(1, 0, 1), ivec3(0, 0, 1)
};

static const std::array<ivec2, 16> uvOffsets = {
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

                if (thisBlock == Block::AIR)
                {
                    continue;
                }

                BlockData thisBlockData = BlockUtils::getBlockData(thisBlock);

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

                        const auto thisTrans = thisBlockData.transparency;
                        const auto neighborTrans = BlockUtils::getBlockData(neighborBlock).transparency;

                        // OPAQUE displays unless neighbor is OPAQUE
                        // SEMI_TRANSPARENT displays no matter what
                        // TRANSPARENT (except AIR) displays unless neighbor is TRANSPARENT (may need to revise this if two different transparent blocks are adjacent)
                        // X_SHAPED displays no matter what
                        if (thisTrans == neighborTrans && (thisTrans == TransparencyType::OPAQUE || thisTrans == TransparencyType::TRANSPARENT))
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
