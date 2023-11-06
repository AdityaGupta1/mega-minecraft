#pragma once

#include "block.hpp"

#define MAX_FEATURES_PER_CHUNK 256

enum class Biome : unsigned char
{
    PLAINS,
    DESERT,
    PURPLE_MUSHROOMS,
    METEORS
};

static constexpr int numBiomes = (int)Biome::METEORS + 1;

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
    SLATE,
    STONE,
    TUFF,
    CALCITE,
    GRANITE,
    MARBLE,
    ANDESITE,
    
    // stratified but placed backwards
    SANDSTONE,

    // eroded
    GRAVEL,
    DIRT
};

static constexpr int numMaterials = (int)Material::DIRT + 1;
static constexpr int numStratifiedMaterials = (int)Material::SANDSTONE + 1;
static constexpr int numForwardMaterials = (int)Material::ANDESITE + 1;
static constexpr int numErodedMaterials = numMaterials - numStratifiedMaterials;

struct MaterialInfo
{
    Block block;
    float thickness;
    float noiseAmplitudeOrTanAngleOfRepose;
    float noiseScaleOrMaxSlope;
};

enum class Feature : unsigned char
{
    NONE,
    SPHERE,
    PURPLE_MUSHROOM
};

static constexpr int numFeatures = (int)Feature::PURPLE_MUSHROOM + 1;

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