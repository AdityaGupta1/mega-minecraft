#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "biomeFuncs.hpp"
#include "featurePlacement.hpp"
#include "util/rng.hpp"

#define BIOME_OVERRIDE Biome::METEORS

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

#pragma region utility functions

template<int size = 16>
__host__ __device__
int posTo2dIndex(const int x, const int z)
{
    return x + size * z;
}

template<int size = 16>
__host__ __device__
int posTo2dIndex(const ivec2 pos)
{
    return posTo2dIndex<size>(pos.x, pos.y);
}

template<int xzSize = 16, int ySize = 384>
__host__ __device__
int posTo3dIndex(const int x, const int y, const int z)
{
    return y + ySize * posTo2dIndex<xzSize>(x, z);
}

template<int xzSize = 16, int ySize = 384>
__host__ __device__
int posTo3dIndex(const ivec3 pos)
{
    return posTo3dIndex<xzSize, ySize>(pos.x, pos.y, pos.z);
}

__host__ __device__ Biome getRandomBiome(const float* columnBiomeWeights, float rand)
{
    for (int i = 0; i < (int)Biome::numBiomes; ++i)
    {
        rand -= columnBiomeWeights[i];
        if (rand <= 0.f)
        {
            return (Biome)i;
        }
    }

    return Biome::PLAINS;
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

__device__ float getBiomeNoise(vec2 pos, float noiseScale, vec2 offset, float smoothstepThreshold, float overallBiomeScale)
{
    return glm::smoothstep(-smoothstepThreshold * overallBiomeScale, smoothstepThreshold * overallBiomeScale, glm::simplex(pos * noiseScale + offset));
}

__global__ void kernGenerateHeightfield(
    float* heightfield,
    float* biomeWeights,
    ivec3 chunkWorldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int z = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = posTo2dIndex(x, z);

    const vec2 worldPos = vec2(chunkWorldBlockPos.x + x, chunkWorldBlockPos.z + z);

    const vec2 noiseOffset = vec2(
        glm::simplex(worldPos * 0.015f + vec2(6839.19f, 1803.34f)),
        glm::simplex(worldPos * 0.015f + vec2(8230.53f, 2042.84f))
    ) * 14.f;
    
    const float overallBiomeScale = 0.4f;
    const vec2 biomeNoisePos = (worldPos + noiseOffset) * overallBiomeScale;

    const float moisture = getBiomeNoise(biomeNoisePos, 0.005f, vec2(1835.32f, 3019.39f), 0.15f, overallBiomeScale);
    const float magic = getBiomeNoise(biomeNoisePos, 0.003f, vec2(5612.35f, 9182.49f), 0.20f, overallBiomeScale);

    float* columnBiomeWeights = biomeWeights + (int)Biome::numBiomes * idx;

    float height = 0.f;
    for (int i = 0; i < (int)Biome::numBiomes; ++i) 
    {
        Biome biome = (Biome)i;

#ifdef BIOME_OVERRIDE
        float weight = (biome == BIOME_OVERRIDE) ? 1.f : 0.f;
#else
        float weight = getBiomeWeight(biome, moisture, magic);
#endif
        if (weight > 0.f)
        {
            height += weight * getHeight((Biome)i, worldPos);
        }

        columnBiomeWeights[i] = weight;
    }
    heightfield[idx] = height;
}

void Chunk::generateHeightfield(
    float* dev_heightfield, 
    float* dev_biomeWeights, 
    cudaStream_t stream)
{
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(2, 2);
    kernGenerateHeightfield<<<blocksPerGrid2d, blockSize2d, 0, stream>>>(
        dev_heightfield,
        dev_biomeWeights,
        this->worldBlockPos
    );

    cudaMemcpyAsync(this->heightfield.data(), dev_heightfield, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(this->biomeWeights.data(), dev_biomeWeights, 256 * (int)Biome::numBiomes * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // used here so cudaMemcpyAsync to this->heightfield finishes before generating own feature placements

    generateOwnFeaturePlacements(); // TODO: maybe move to a separate step

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
                // +/- x
                int xIn, xOut;
                calculateEdgeIndices(offsetX, xIn, xOut);

                for (int z = 0; z < 16; ++z)
                {
                    chunkPtr->gatheredHeightfield[posTo2dIndex<18>(xOut, z + 1)] = neighborPtr->heightfield[posTo2dIndex(xIn, z)];
                }
            }
            else
            {
                // +/- z
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

__global__ void kernGenerateLayers(
    float* heightfield,
    float* layers,
    float* biomeWeights,
    ivec3 chunkWorldBlockPos)
{
    __shared__ float shared_heightfield[18 * 18];

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int z = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = posTo2dIndex(x, z);

    const vec2 worldPos = vec2(chunkWorldBlockPos.x + x, chunkWorldBlockPos.z + z);

    shared_heightfield[idx] = heightfield[idx];
    const int idx2 = idx + 256;
    if (idx2 < 18 * 18)
    {
        shared_heightfield[idx2] = heightfield[idx2];
    }

    __syncthreads();

    const ivec2 pos18 = ivec2(x + 1, z + 1);
    const float maxHeight = shared_heightfield[posTo2dIndex<18>(pos18)];

    float slope = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float neighborHeight = shared_heightfield[posTo2dIndex<18>(pos18 + dev_dirVecs2d[i])];
        slope = max(slope, abs(neighborHeight - maxHeight));
    }

    float* columnLayers = layers + (int)Material::numMaterials * idx;

    float height = 0;
    #pragma unroll
    for (int layerIdx = 0; layerIdx < numStratifiedMaterials; ++layerIdx)
    {
        columnLayers[layerIdx] = height;

        if (height > maxHeight)
        {
            continue;
        }

        if (layerIdx != numStratifiedMaterials - 1)
        {
            const auto& materialInfo = dev_materialInfos[layerIdx];
            vec2 noisePos = worldPos * materialInfo.noiseScaleOrMaxSlope + vec2(layerIdx * 5283.64f);
            height += max(0.f, materialInfo.thickness + materialInfo.noiseAmplitudeOrTanAngleOfRepose * fbm(noisePos));
        }
    }

    height = maxHeight;
    for (int layerIdx = (int)Material::numMaterials - 1; layerIdx >= numStratifiedMaterials; --layerIdx)
    {
        const auto& materialInfo = dev_materialInfos[layerIdx];

        float layerHeight;
        if (slope > materialInfo.noiseScaleOrMaxSlope)
        {
            layerHeight = 0;
        }
        else
        {
            layerHeight = materialInfo.thickness * ((materialInfo.noiseScaleOrMaxSlope - slope) / materialInfo.noiseScaleOrMaxSlope);
        }

        height -= layerHeight;
        columnLayers[layerIdx] = height;
    }
}

void Chunk::generateLayers(float* dev_heightfield, float* dev_layers, float* dev_biomeWeights, cudaStream_t stream)
{
    cudaMemcpyAsync(dev_heightfield, this->gatheredHeightfield.data(), 18 * 18 * sizeof(float), cudaMemcpyHostToDevice, stream);
    this->gatheredHeightfield.clear();
    cudaMemcpyAsync(dev_biomeWeights, this->biomeWeights.data(), 256 * (int)Biome::numBiomes * sizeof(float), cudaMemcpyHostToDevice, stream);

    const dim3 blockSize2d(16, 16);
    const dim3 blocksPerGrid2d(1, 1);
    kernGenerateLayers<<<blocksPerGrid2d, blockSize2d, 0, stream>>>(
        dev_heightfield,
        dev_layers,
        dev_biomeWeights,
        this->worldBlockPos
    );

    cudaMemcpyAsync(this->layers.data(), dev_layers, 256 * (int)Material::numMaterials * sizeof(float), cudaMemcpyDeviceToHost, stream);

    CudaUtils::checkCUDAError("Chunk::generateLayers() failed");
}

#pragma endregion

#pragma region erosion

__global__ void kernDoErosion(float* gatheredLayers, int layer)
{
    __shared__ float shared_layer[34 * 34]; // 32x32 with 1 padding
    __shared__ bool didChange;

    const int localX = threadIdx.x;
    const int localZ = threadIdx.y;
    const int localIdx = posTo2dIndex<32>(localX, localZ);
    const int blockStartX = (blockIdx.x * blockDim.x);
    const int blockStartZ = (blockIdx.y * blockDim.y);
    const int globalX = blockStartX + localX;
    const int globalZ = blockStartZ + localZ;

    const int sharedLayerIdx = posTo2dIndex<34>(localX + 1, localZ + 1);
    const int gatheredLayersIdx = posTo3dIndex<ZONE_SIZE * 2 * 16, (int)Material::numMaterials>(globalX, layer, globalZ);

    float thisHeight = gatheredLayers[gatheredLayersIdx];
    shared_layer[sharedLayerIdx] = thisHeight;

    // TODO: load border values

    __syncthreads();

    // TODO: actually do erosion lol
    //       don't overwrite shared memory, just change thisHeight
    thisHeight += 100.f;

    gatheredLayers[gatheredLayersIdx] = thisHeight;
}

void copyLayers(Zone* zonePtr, float* gatheredLayers, bool toGatheredLayers)
{
    int maxDim = toGatheredLayers ? ZONE_SIZE * 2 : ZONE_SIZE;

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

            for (int blockZ = 0; blockZ < 16; ++blockZ)
            {
                for (int blockX = 0; blockX < 16; ++blockX)
                {
                    auto srcLayers = chunkPtr->layers[posTo2dIndex(blockX, blockZ)];
                    auto dstLayers = gatheredLayers + posTo3dIndex<ZONE_SIZE * 2 * 16, (int)Material::numMaterials>(chunkBlockPos.x + blockX, 0, chunkBlockPos.y + blockZ);
                    if (!toGatheredLayers)
                    {
                        std::swap(srcLayers, dstLayers);
                    }
                    std::memcpy(dstLayers, srcLayers, (int)Material::numMaterials * sizeof(float));
                }
            }
        }
    }
}

void Chunk::erodeZone(Zone* zonePtr, float* dev_gatheredLayers, cudaStream_t stream)
{
    std::array<float, ZONE_SIZE * ZONE_SIZE * 4 * 256 * (int)Material::numMaterials + 1> gatheredLayers;
    copyLayers(zonePtr, gatheredLayers.data(), true);
    zonePtr->gatheredChunks.clear();

    int gatheredLayersSizeBytes = gatheredLayers.size() * sizeof(float);
    cudaMemcpyAsync(dev_gatheredLayers, gatheredLayers.data(), gatheredLayersSizeBytes, cudaMemcpyHostToDevice, stream);

    const dim3 blockSize2d(32, 32);
    const int blocksPerGrid = (ZONE_SIZE * 2 * 16) / 32; // = ZONE_SIZE but writing it out for clarity
    const dim3 blocksPerGrid2d(blocksPerGrid, blocksPerGrid);
    // TODO: multiple kernel executions until no more changes, and then move to next material
    kernDoErosion<<<blocksPerGrid2d, blockSize2d, 0, stream>>>(dev_gatheredLayers, (int)Material::DIRT);

    cudaMemcpyAsync(gatheredLayers.data(), dev_gatheredLayers, gatheredLayersSizeBytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // all data needs to be copied back to gatheredLayers before calling copyLayers()
                                   // explicit synchronization here may not be necessary (seems to work without it) but it gives peace of mind

    copyLayers(zonePtr, gatheredLayers.data(), false);

    for (const auto& chunkPtr : zonePtr->chunks)
    {
        chunkPtr->setState(ChunkState::NEEDS_GATHER_FEATURE_PLACEMENTS);
    }
}

#pragma endregion

#pragma region feature placements

void Chunk::generateOwnFeaturePlacements()
{
    for (int localZ = 0; localZ < 16; ++localZ)
    {
        for (int localX = 0; localX < 16; ++localX)
        {
            const int idx2d = posTo2dIndex(localX, localZ);

            const auto& columnBiomeWeights = biomeWeights[idx2d];

            const ivec3 worldBlockPos = this->worldBlockPos + ivec3(localX, heightfield[idx2d], localZ);
            auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z, 7); // arbitrary w so this rng is different than heightfield rng
            thrust::uniform_real_distribution<float> u01(0, 1);

            Biome biome = getRandomBiome(columnBiomeWeights, u01(rng));
            const auto& featureGens = BiomeUtils::getBiomeFeatureGens(biome);

            Feature feature = Feature::NONE;
            float rand = u01(rng);
            for (int i = 0; i < featureGens.size(); ++i)
            {
                const auto& featureGen = featureGens[i];

                rand -= featureGen.chancePerBlock;
                if (rand <= 0.f)
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
    float* layers,
    float* biomeWeights,
    FeaturePlacement* dev_featurePlacements,
    int numFeaturePlacements,
    ivec2 featureHeightBounds,
    ivec3 chunkWorldBlockPos)
{
    __shared__ float shared_layersAndHeight[(int)Material::numMaterials + 1];
    __shared__ float shared_biomeWeights[(int)Biome::numBiomes];

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posTo3dIndex(x, y, z);
    const int idx2d = posTo2dIndex(x, z);

    if (y < (int)Material::numMaterials)
    {
        const float* columnLayers = layers + (int)Material::numMaterials * idx2d;
        shared_layersAndHeight[y] = columnLayers[y];
    }
    else if (y == (int)Material::numMaterials)
    {
        shared_layersAndHeight[y] = heightfield[idx2d];
    }
    else
    {
        const int biomeWeightIdx = y - (int)Material::numMaterials - 1;
        if (biomeWeightIdx < (int)Biome::numBiomes)
        {
            const float* columnBiomeWeights = biomeWeights + (int)Biome::numBiomes * idx2d;
            shared_biomeWeights[biomeWeightIdx] = columnBiomeWeights[biomeWeightIdx];
        }
    }

    __syncthreads();

    const float height = heightfield[idx2d];

    const ivec3 worldBlockPos = chunkWorldBlockPos + ivec3(x, y, z);
    auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Block block = Block::AIR;
    if (y < height)
    {
        int layerIdxStart;
        if (y >= shared_layersAndHeight[numStratifiedMaterials])
        {
            layerIdxStart = numStratifiedMaterials;
        }
        else
        {
            layerIdxStart = 0;
        }

        int thisLayerIdx = -1;
        bool isTopBlock;
        for (int layerIdx = layerIdxStart; layerIdx < (int)Material::numMaterials + 1; ++layerIdx)
        {
            if (y < shared_layersAndHeight[layerIdx])
            {
                thisLayerIdx = layerIdx - 1;
                isTopBlock = shared_layersAndHeight[layerIdx] - y < 1.f;
                break;
            }
        }

        block = dev_materialInfos[thisLayerIdx].block;
        if (isTopBlock)
        {
            // TODO: use biome-specific options (e.g. dirt to grass or mycelium depending on biome)
            // also need to support replacing other things (maybe just make top block a generic block that always gets replaced?

            if (block == Block::DIRT)
            {
                block = Block::GRASS;
            }
        }
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
    float* dev_layers,
    float* dev_biomeWeights, 
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

    int numFeaturePlacements = this->gatheredFeaturePlacements.size();
    cudaMemcpyAsync(dev_featurePlacements, this->gatheredFeaturePlacements.data(), numFeaturePlacements * sizeof(FeaturePlacement), cudaMemcpyHostToDevice, stream);
    this->gatheredFeaturePlacements.clear();

    cudaMemcpyAsync(dev_heightfield, this->heightfield.data(), 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_layers, this->layers.data(), 256 * (int)Material::numMaterials * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_biomeWeights, this->biomeWeights.data(), 256 * (int)Biome::numBiomes * sizeof(float), cudaMemcpyHostToDevice, stream);

    const dim3 blockSize3d(1, 384, 1);
    const dim3 blocksPerGrid3d(16, 1, 16);
    kernFill<<<blocksPerGrid3d, blockSize3d, 0, stream>>>(
        dev_blocks, 
        dev_heightfield,
        dev_layers,
        dev_biomeWeights,
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

                        if (neighborBlock != Block::AIR) // TODO: this will get more complicated with transparent and non-cube blocks
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
