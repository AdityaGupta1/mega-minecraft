#pragma once

#include "block.hpp"

#define MAX_FEATURES_PER_CHUNK 128

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

enum class Feature : unsigned char
{
    SPHERE
};

struct FeatureGen
{
    Feature feature;
    float chancePerBlock;
};

struct FeaturePlacement
{
    Feature feature;
    glm::ivec3 pos;
};

namespace BiomeUtils
{
    void init(); // implemented in biomeFuncs.hpp (included only by chunk.cu) so constant memory can live there
}