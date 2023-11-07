#pragma once

#include "block.hpp"

#define MAX_FEATURES_PER_CHUNK 256

enum class Biome : unsigned char
{
    JUNGLE,
    RED_DESERT,
    PURPLE_MUSHROOMS,
    CRYSTALS,
    OASIS,
    DESERT,
    PLAINS,
    MOUNTAINS
};

static constexpr int numBiomes = (int)Biome::MOUNTAINS + 1;

struct BiomeBlocks
{
    Block grassBlock{ Block::GRASS };
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
    RED_SANDSTONE,
    SANDSTONE,

    // eroded
    GRAVEL,
    CLAY,
    MUD,
    DIRT,
    RED_SAND,
    SAND
};

static constexpr int numMaterials = (int)Material::SAND + 1;
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
    PURPLE_MUSHROOM,
    RAFFLESIA,
    SMALL_JUNGLE_TREE
    //LARGE_JUNGLE_TREE
    //PALM_TREE
    //POND (not sure if this should go here or somewhere else)
    //CACTUS
    //CRYSTAL
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