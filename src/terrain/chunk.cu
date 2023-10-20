#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "cuda/cudaUtils.hpp"
#include "biomeFuncs.hpp"
#include "featurePlacement.hpp"
#include "util/rng.hpp"

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos), worldBlockPos(worldChunkPos.x * 16, 0, worldChunkPos.y * 16)
{}

ChunkState Chunk::getState()
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

__host__ __device__
int posToIndex(const int x, const int y, const int z)
{
    return x + 16 * z + 256 * y;
}

__host__ __device__
int posToIndex(const ivec3 pos)
{
    return posToIndex(pos.x, pos.y, pos.z);
}

__host__ __device__
int posToIndex(const int x, const int z)
{
    return x + 16 * z;
}

__host__ __device__
int posToIndex(const ivec2 pos)
{
    return posToIndex(pos.x, pos.y);
}

__global__ void kernGenerateHeightfield(
    unsigned char* heightfield,
    float* biomeWeights,
    ivec3 worldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int z = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = posToIndex(x, z);

    const vec2 worldPos = vec2(worldBlockPos.x + x, worldBlockPos.z + z);

    const vec2 noiseOffset = vec2(
        glm::simplex(worldPos * 0.015f + vec2(6839.19f, 1803.34f)),
        glm::simplex(worldPos * 0.015f + vec2(8230.53f, 2042.84f))
    ) * 14.f;
    
    const float overallBiomeScale = 0.4f;
    const vec2 biomeNoisePos = (worldPos + noiseOffset) * overallBiomeScale;

    const float moisture = glm::smoothstep(-0.15f * overallBiomeScale, 0.15f * overallBiomeScale, glm::simplex(biomeNoisePos * 0.005f + vec2(1835.32f, 3019.39f)));
    const float magic = glm::smoothstep(-0.2f * overallBiomeScale, 0.2f * overallBiomeScale, glm::simplex(biomeNoisePos * 0.003f + vec2(5612.35f, 9182.49f)));

    float* columnBiomeWeights = biomeWeights + (int)Biome::numBiomes * idx;

    float height = 0.f;
    for (int i = 0; i < (int)Biome::numBiomes; ++i) 
    {
        Biome biome = (Biome)i;

        float weight = getBiomeWeight(biome, moisture, magic);
        if (weight > 0.f)
        {
            height += weight * getHeight((Biome)i, worldPos);
        }

        columnBiomeWeights[i] = weight;
    }
    heightfield[idx] = (int)height;
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
}

__global__ void kernFill(
    Block* blocks, 
    unsigned char* heightfield, 
    float* biomeWeights, 
    FeaturePlacement* dev_featurePlacements, 
    int numFeaturePlacements,
    ivec2 featureHeightBounds,
    ivec3 chunkWorldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posToIndex(x, y, z);

    const int idx2d = posToIndex(x, z);
    // TODO: use shared memory to load heightfield
    // right now, each thread block covers exactly one column, so shared memory should have to load only one height value and one set of biome weights
    const unsigned char height = heightfield[idx2d];
    const float* columnBiomeWeights = biomeWeights + (int)Biome::numBiomes * idx2d;

    const ivec3 worldBlockPos = chunkWorldBlockPos + ivec3(x, y, z);
    auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z);
    thrust::uniform_real_distribution<float> u01(0, 1);

    BiomeBlocks biomeBlocks = dev_biomeBlocks[(int)getRandomBiome(columnBiomeWeights, u01(rng))];

    Block block = biomeBlocks.blockStone;
    if (y > height)
    {
        block = Block::AIR;
    }
    else if (y == height)
    {
        block = biomeBlocks.blockTop;
    }
    else if (y >= height - 3)
    {
        block = biomeBlocks.blockMid;
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

void Chunk::generateOwnFeaturePlacements()
{
    for (int localZ = 0; localZ < 16; ++localZ)
    {
        for (int localX = 0; localX < 16; ++localX)
        {
            const int idx2d = posToIndex(localX, localZ);

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
                this->featurePlacements.push_back({feature, worldBlockPos});
            }
        }
    }

    // this probably won't include decorators (single block/column things) since those can be done on the CPU at the end of Chunk::fill()
}

void Chunk::generateHeightfield(unsigned char* dev_heightfield, float* dev_biomeWeights)
{
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(2, 2);

    kernGenerateHeightfield<<<blocksPerGrid2d, blockSize2d>>>(
        dev_heightfield,
        dev_biomeWeights,
        this->worldBlockPos
    );
    CudaUtils::checkCUDAError("kernGenerateHeightfield failed");

    cudaMemcpy(this->heightfield.data(), dev_heightfield, 256 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->biomeWeights.data(), dev_biomeWeights, 256 * (int)Biome::numBiomes * sizeof(float), cudaMemcpyDeviceToHost);
    CudaUtils::checkCUDAError("cudaMemcpy to host failed");

    generateOwnFeaturePlacements();
}

bool Chunk::otherChunkGatherFeaturePlacements(Chunk* chunkPtr, Chunk* (&neighborChunks)[9][9], int centerX, int centerZ)
{
    chunkPtr->gatheredFeaturePlacements.clear();

    for (int offsetZ = -2; offsetZ <= 2; ++offsetZ)
    {
        for (int offsetX = -2; offsetX <= 2; ++offsetX)
        {
            const auto& neighborPtr = neighborChunks[centerZ + offsetZ][centerX + offsetX];

            if (neighborPtr == nullptr)
            {
                chunkPtr->gatheredFeaturePlacements.clear();
                return false;
            }

            for (const auto& neighborFeaturePlacement : neighborPtr->featurePlacements)
            {
                chunkPtr->gatheredFeaturePlacements.push_back(neighborFeaturePlacement);
            }
        }
    }

    return true;
}

void Chunk::gatherFeaturePlacements()
{
    Chunk* neighborChunks[9][9] = {};

    // Flood fill neighborChunks (connected chunks that exist and have feature placements).
    // If a chunk's 5x5 area is ready to go, it will be reached by flood fill (since this chunk is contained in that area).
    // By the contrapositive, if a chunk was not reached by flood fill, its 5x5 area is not ready to go.
    std::queue<Chunk*> chunks;
    std::unordered_set<Chunk*> visitedChunks;
    chunks.push(this);
    while (!chunks.empty())
    {
        auto& chunkPtr = chunks.front();
        visitedChunks.insert(chunkPtr);

        if (chunkPtr->getState() < ChunkState::HAS_HEIGHTFIELD_AND_FEATURE_PLACEMENTS)
        {
            continue;
        }

        ivec2 neighborChunksIdx = chunkPtr->worldChunkPos - this->worldChunkPos + ivec2(4, 4);
        neighborChunks[neighborChunksIdx.y][neighborChunksIdx.x] = chunkPtr;

        for (const auto& neighborPtr : chunkPtr->neighbors)
        {
            if (neighborPtr == nullptr || visitedChunks.find(neighborPtr) != visitedChunks.end())
            {
                continue;
            }

            const ivec2 dist = abs(neighborPtr->worldChunkPos - this->worldChunkPos);
            if (max(dist.x, dist.y) > 4)
            {
                continue;
            }

            chunks.push(neighborPtr);
        }

        chunks.pop();
    }

    for (int centerZ = 2; centerZ < 7; ++centerZ)
    {
        for (int centerX = 2; centerX < 7; ++centerX)
        {
            const auto& chunkPtr = neighborChunks[centerZ][centerX];

            if (chunkPtr == nullptr || chunkPtr->getState() != ChunkState::HAS_HEIGHTFIELD_AND_FEATURE_PLACEMENTS)
            {
                continue;
            }

            bool isReady = otherChunkGatherFeaturePlacements(chunkPtr, neighborChunks, centerX, centerZ);
            if (isReady)
            {
                chunkPtr->setState(ChunkState::READY_TO_FILL);
            }
        }
    }
}

void Chunk::fill(Block* dev_blocks, unsigned char* dev_heightfield, float* dev_biomeWeights, FeaturePlacement* dev_featurePlacements)
{
    ivec2 allFeaturesHeightBounds = ivec2(256, -1);
    for (const auto& featurePlacement : this->gatheredFeaturePlacements)
    {
        const auto& featureHeightBounds = BiomeUtils::getFeatureHeightBounds(featurePlacement.feature);
        const ivec2 thisFeatureHeightBounds = ivec2(featurePlacement.pos.y) + featureHeightBounds;
        allFeaturesHeightBounds[0] = min(allFeaturesHeightBounds[0], thisFeatureHeightBounds[0]);
        allFeaturesHeightBounds[1] = max(allFeaturesHeightBounds[1], thisFeatureHeightBounds[1]);
    }

    int numFeaturePlacements = this->gatheredFeaturePlacements.size();
    cudaMemcpy(dev_featurePlacements, this->gatheredFeaturePlacements.data(), numFeaturePlacements * sizeof(FeaturePlacement), cudaMemcpyHostToDevice);
    this->gatheredFeaturePlacements.clear();

    cudaMemcpy(dev_heightfield, this->heightfield.data(), 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_biomeWeights, this->biomeWeights.data(), 256 * (int)Biome::numBiomes * sizeof(float), cudaMemcpyHostToDevice);
    CudaUtils::checkCUDAError("cudaMemcpy to device failed");

    const dim3 blockSize3d(1, 256, 1);
    const dim3 blocksPerGrid3d(16, 1, 16);

    kernFill<<<blocksPerGrid3d, blockSize3d>>>(
        dev_blocks, 
        dev_heightfield,
        dev_biomeWeights,
        dev_featurePlacements,
        numFeaturePlacements,
        allFeaturesHeightBounds,
        this->worldBlockPos
    );
    CudaUtils::checkCUDAError("kernFill failed");
    
    cudaMemcpy(this->blocks.data(), dev_blocks, 65536 * sizeof(Block), cudaMemcpyDeviceToHost);
    CudaUtils::checkCUDAError("cudaMemcpy to host failed");
}

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

    for (int y = 0; y < 256; ++y)
    {
        for (int z = 0; z < 16; ++z)
        {
            for (int x = 0; x < 16; ++x)
            {
                ivec3 thisPos = ivec3(x, y, z);
                Block thisBlock = blocks[posToIndex(thisPos)];

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

                    if (neighborPos.y >= 0 && neighborPos.y < 256)
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

                        neighborBlock = neighborPosChunk->blocks[posToIndex(neighborPos)];

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