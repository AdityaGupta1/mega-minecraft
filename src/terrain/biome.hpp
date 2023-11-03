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
    BLACKSTONE,
    DEEPSLATE,
    STONE,
    TUFF,
    CALCITE,
    GRANITE,
    MARBLE,
    ANDESITE,

    // eroded
    DIRT,

    numMaterials
};

const int numStratifiedMaterials = (int)Material::DIRT;

struct MaterialInfo
{
    Block block;
    float thickness;
    float noiseAmplitudeOrAngleOfRepose;
    float noiseScaleOrMaximumSlope;
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