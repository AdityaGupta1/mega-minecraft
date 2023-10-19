#pragma once

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
    void init(); // implemented in biomeFuncs.hpp (included only by chunk.cu) so constant memory can live there
}