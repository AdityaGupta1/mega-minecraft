#pragma once

#include "block.hpp"

#define MAX_FEATURES_PER_CHUNK 256

enum class Biome : unsigned char
{
    PLAINS,
    DESERT,
    PURPLE_MUSHROOMS,
    METEORS,

    numBiomes
};

struct BiomeBlocks
{
    Block blockTop;
    Block blockMid;
    Block blockStone;
};

enum class Material : unsigned char
{
    // stratified
    //DEEPSLATE,
    STONE,
    //BLACKSTONE,
    //TUFF,
    //CALCITE,
    //ANDESITE,
    //MARBLE,

    // eroded
    DIRT,

    numMaterials
};

const int numStratifiedMaterials = 1;

struct MaterialInfo
{
    float thickness;
    float roughnessOrAngleOfRepose;
    float maximumSlope;
};

enum class Feature : unsigned char
{
    NONE,
    SPHERE,
    PURPLE_MUSHROOM,

    numFeatures
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
    using namespace glm;

    void init(); // implemented in biomeFuncs.hpp (included only by chunk.cu) so constant memory can live there

    std::vector<FeatureGen>& getBiomeFeatureGens(Biome biome);

    ivec2 getFeatureHeightBounds(Feature feature);
}