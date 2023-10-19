#pragma once

#include "cuda/cuda_utils.hpp"
#include "block.hpp"

enum class Biome : unsigned char
{
    PLAINS,
    DESERT,
    MUSHROOMS,
    METEORS,

    // not an actual biome, just for counting
    numBiomes
};

struct BiomeBlocks
{
    Block blockTop;
    Block blockMid;
    Block blockStone;
};

namespace BiomeUtils
{
    void init(); // implemented in chunk.cu so constant memory can live there
}