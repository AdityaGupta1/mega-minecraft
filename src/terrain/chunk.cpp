#include "chunk.hpp"

Chunk::Chunk(ivec2 worldChunkPos)
    : worldChunkPos(worldChunkPos), heightfield(), blocks()
{
}

void Chunk::dummyFill()
{
    for (int y = 0; y < 64; ++y)
    {
        for (int z = 0; z < 16; ++z)
        {
            for (int x = 0; x < 16; ++x)
            {
                this->blocks[x + 16 * z + 256 * y] = Block::STONE;
            }
        }
    }
}