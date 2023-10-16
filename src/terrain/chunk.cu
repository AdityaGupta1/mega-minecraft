#include "chunk.hpp"

#include "rendering/structs.hpp"
#include "rendering/renderingUtils.hpp"
#include "util/enums.hpp"
#include "cuda/cuda_utils.hpp"
#include <glm/gtc/noise.hpp>

#include <iostream>

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

void Chunk::dummyFill()
{
    for (int z = 0; z < 16; ++z)
    {
        for (int x = 0; x < 16; ++x)
        {
            int height = 48 + (x + z) / 2;
            if ((x + z) % 4 == 1)
            {
                height += 10;
            }
            for (int y = 0; y < height; ++y)
            {
                Block block = Block::STONE;
                if (y == height - 1)
                {
                    block = Block::GRASS;
                }
                else if (y < height - 1 && y >= height - 4)
                {
                    block = Block::DIRT;
                }

                this->blocks[posToIndex(x, y, z)] = block;
            }
        }
    }
}

__device__
float dummyNoise(vec2 pos)
{
    pos *= 0.02f;

    float fbm = 0.f;
    float amplitude = 1.f;
    for (int i = 0; i < 4; ++i)
    {
        amplitude *= 0.5f;
        pos *= 2.f;
        fbm += amplitude * glm::simplex(pos);
    }

    return 64.f + 6.f * fbm;
}

__global__ void kernDummyGenerateHeightfield(unsigned char* heightfield, ivec2 worldBlockPos)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int z = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = posToIndex(x, z);

    const vec2 worldPos = vec2(worldBlockPos.x + x, worldBlockPos.y + z);

    int height = (int)dummyNoise(worldPos);

    heightfield[idx] = height;
}

__global__ void kernDummyFill(Block* blocks, unsigned char* heightfield)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    const int idx = posToIndex(x, y, z);

    const unsigned char height = heightfield[posToIndex(x, z)]; // TODO: when implementing for real, use shared memory to load heightfield

    Block block = Block::STONE;
    if (y > height)
    {
        block = Block::AIR;
    }
    else if (y == height)
    {
        block = Block::GRASS;
    }
    else if (y >= height - 3)
    {
        block = Block::DIRT;
    }

    blocks[idx] = block;
}

void Chunk::dummyFillCUDA(Block* dev_blocks, unsigned char* dev_heightfield)
{
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(2, 2);

    const dim3 blockSize3d(1, 256, 1);
    const dim3 blocksPerGrid3d(16, 1, 16);

    cudaEvent_t start, mid, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&mid);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernDummyGenerateHeightfield<<<blocksPerGrid2d, blockSize2d>>>(dev_heightfield, this->worldChunkPos * 16);
    CudaUtils::checkCUDAError("kern generate heightfield failed");
    cudaEventRecord(mid);

    cudaDeviceSynchronize();

    // TODO: when implementing for real, the two kernels will happen separately; will probably need to copy heightfield back to GPU before running this kernel
    kernDummyFill<<<blocksPerGrid3d, blockSize3d>>>(dev_blocks, dev_heightfield);
    CudaUtils::checkCUDAError("kern fill failed");
    
    cudaMemcpy(this->blocks.data(), dev_blocks, 65536 * sizeof(Block), cudaMemcpyDeviceToHost);
    CudaUtils::checkCUDAError("cudaMemcpy failed");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "full ms elapsed: " << milliseconds << std::endl;
    //cudaEventElapsedTime(&milliseconds, start, mid);
    //std::cout << "mid ms elapsed: " << milliseconds << std::endl;
    //std::cout << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(mid);
    cudaEventDestroy(stop);
}

static const std::array<vec3, 24> directionVertPositions = {
    vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1),
    vec3(1, 0, 1), vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1),
    vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0),
    vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0),
    vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0), vec3(0, 1, 0),
    vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1)
};

static const std::array<vec2, 4> uvOffsets = {
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

                for (int i = 0; i < 6; ++i)
                {
                    const auto& direction = DirectionEnums::dirVecs[i];
                    const ivec3 newPos = thisPos + direction;
                    if (newPos.x < 0 || newPos.x >= 16 || newPos.z < 0 || newPos.z >= 16 || newPos.y < 0 || newPos.y >= 256)
                    {
                        continue;
                    }

                    Block neighborBlock = blocks[posToIndex(newPos)];
                    if (neighborBlock != Block::AIR)
                    {
                        continue;
                    }

                    int idx1 = verts.size();

                    vec2 uvStart;
                    switch (direction.y)
                    {
                    case 1:
                        uvStart = thisBlockData.uvs.top;
                        break;
                    case -1:
                        uvStart = thisBlockData.uvs.bottom;
                        break;
                    case 0:
                        uvStart = thisBlockData.uvs.side;
                        break;
                    }

                    for (int j = 0; j < 4; ++j)
                    {
                        verts.emplace_back();
                        Vertex& vert = verts.back();
                        vert.pos = vec3(thisPos) + directionVertPositions[i * 4 + j];
                        vert.uv = uvStart + uvOffsets[j];
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