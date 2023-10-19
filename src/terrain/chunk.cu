#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "cuda/cuda_utils.hpp"
#include <glm/gtc/noise.hpp>

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos)
{
    std::fill_n(heightfield.begin(), 256, 0);
    std::fill_n(blocks.begin(), 65536, Block::AIR);
}

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

__device__ float fbm(vec2 pos)
{
    float fbm = 0.f;
    float amplitude = 1.f;
    for (int i = 0; i < 4; ++i)
    {
        amplitude *= 0.5f;
        pos *= 2.f;
        fbm += amplitude * glm::simplex(pos);
    }
    return fbm;
}

__device__ float getHeight(vec2 pos, Biome biome)
{
    switch (biome)
    {
    case Biome::PLAINS:
        return 72.f + 12.f * fbm(pos * 0.01f);
    case Biome::DESERT:
        return 64.f + 5.f * fbm(pos * 0.005f);
    }
}

__global__ void kernDummyGenerateHeightfield(unsigned char* heightfield, float* biomeWeights, ivec2 worldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int z = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = posToIndex(x, z);

    const vec2 worldPos = vec2(worldBlockPos.x + x, worldBlockPos.y + z);

    // biomes
    float* biomeWeightsStart = biomeWeights + (int)Biome::numBiomes * idx;
    
    const float moisture = glm::smoothstep(0.45f, 0.55f, glm::simplex(worldPos * 0.005f));
   
    biomeWeightsStart[(int)Biome::PLAINS] = moisture;
    biomeWeightsStart[(int)Biome::DESERT] = 1.f - moisture;

    // height
    float height = 0.f;
    for (int i = 0; i < (int)Biome::numBiomes; ++i) 
    {
        float weight = biomeWeightsStart[i];
        if (weight > 0.f)
        {
            height += weight * getHeight(worldPos, (Biome)i);
        }
    }
    heightfield[idx] = (int)height;
}

__constant__ BiomeBlocks dev_biomeBlocks[(int)Biome::numBiomes];

void BiomeUtils::init()
{
    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[(int)Biome::numBiomes];

    host_biomeBlocks[(int)Biome::PLAINS] = { Block::GRASS, Block::DIRT, Block::STONE };
    host_biomeBlocks[(int)Biome::DESERT] = { Block::SAND, Block::SAND, Block::STONE };

    cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, (int)Biome::numBiomes * sizeof(BiomeBlocks));

    delete[] host_biomeBlocks;
}

__host__ __device__ inline unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x)
{
    int h = hash(x);
    return thrust::default_random_engine(h);
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x, int y, int z)
{
    int h = hash((1 << 31) | (x << 22) | y) ^ hash(z);
    return thrust::default_random_engine(h);
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int x, int y, int z, int w)
{
    int h = hash((1 << 31) | (x << 22) | (y << 11) | w) ^ hash(z);
    return thrust::default_random_engine(h);
}

__global__ void kernDummyFill(Block* blocks, unsigned char* heightfield, float* biomeWeights, ivec2 worldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posToIndex(x, y, z);

    const int idx2d = posToIndex(x, z);
    const unsigned char height = heightfield[idx2d]; // TODO: when implementing for real, use shared memory to load heightfield
    const float* biomeWeightsStart = biomeWeights + (int)Biome::numBiomes * idx2d;

    auto rng = makeSeededRandomEngine(worldBlockPos.x + x, y, worldBlockPos.y + z);
    thrust::uniform_real_distribution<float> u01(0, 1);

    BiomeBlocks biomeBlocks;
    float rand = u01(rng);
    for (int i = 0; i < (int)Biome::numBiomes; ++i)
    {
        rand -= biomeWeightsStart[i];
        if (rand <= 0.f)
        {
            biomeBlocks = dev_biomeBlocks[i];
            break;
        }
    }

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

    blocks[idx] = block;
}

void Chunk::dummyFillCUDA(Block* dev_blocks, unsigned char* dev_heightfield, float* dev_biomeWeights)
{
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(2, 2);

    const dim3 blockSize3d(1, 256, 1);
    const dim3 blocksPerGrid3d(16, 1, 16);

    kernDummyGenerateHeightfield<<<blocksPerGrid2d, blockSize2d>>>(
        dev_heightfield,
        dev_biomeWeights,
        this->worldChunkPos * 16
    );
    CudaUtils::checkCUDAError("kern generate heightfield failed");

    cudaDeviceSynchronize();

    // TODO: when implementing for real, the two kernels will happen separately; will probably need to copy heightfield back to GPU before running this kernel
    kernDummyFill<<<blocksPerGrid3d, blockSize3d>>>(
        dev_blocks, 
        dev_heightfield,
        dev_biomeWeights,
        this->worldChunkPos * 16
    );
    CudaUtils::checkCUDAError("kern fill failed");
    
    cudaMemcpy(this->blocks.data(), dev_blocks, 65536 * sizeof(Block), cudaMemcpyDeviceToHost);
    CudaUtils::checkCUDAError("cudaMemcpy failed");

    cudaDeviceSynchronize();
}

static const std::array<vec3, 24> directionVertPositions = {
    vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1),
    vec3(1, 0, 1), vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1),
    vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0),
    vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0),
    vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0), vec3(0, 1, 0),
    vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1)
};

static const std::array<vec2, 16> uvOffsets = {
    vec2(0, 0), vec2(0.0625f, 0), vec2(0.0625f, 0.0625f), vec2(0, 0.0625f)
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
                        auto rng = makeSeededRandomEngine(worldChunkPos.x * 16 + thisPos.x, thisPos.y, worldChunkPos.y * 16 + thisPos.z, dirIdx);
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

                        vert.pos = vec3(thisPos) + directionVertPositions[dirIdx * 4 + j];

                        vec2 uvOffset = uvOffsets[(uvStartIdx + j) % 4];
                        if (uvFlipIdx != -1)
                        {
                            if (uvFlipIdx & 1)
                            {
                                uvOffset.x = 0.0625f - uvOffset.x;
                            }
                            if (uvFlipIdx & 2)
                            {
                                uvOffset.y = 0.0625f - uvOffset.y;
                            }
                        }
                        vert.uv = sideUv.uv + uvOffset;
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