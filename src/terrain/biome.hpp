#pragma once

#include "block.hpp"

#define MAX_GATHERED_FEATURES_PER_CHUNK 1024

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

    // TINY_PURPLE_MUSHROOM
    // SMALL_PURPLE_MUSHROOM
    PURPLE_MUSHROOM,

    RAFFLESIA,
    LARGE_JUNGLE_TREE,
    SMALL_JUNGLE_TREE,
    TINY_JUNGLE_TREE,

    //PALM_TREE
    
    CACTUS
    
    //POND (not sure if this should go here or somewhere else)
    //CRYSTAL
};

static constexpr int numFeatures = (int)Feature::CACTUS + 1;

struct FeatureGenTopLayer
{
    Material material;
    float minThickness;
};

struct FeatureGen
{
    Feature feature;
    int gridCellSize;
    int gridCellPadding;
    float chancePerGridCell;
    std::vector<FeatureGenTopLayer> possibleTopLayers;
};

struct FeaturePlacement
{
    Feature feature;
    glm::ivec3 pos;
};

namespace BiomeUtils
{
    using namespace glm;

    void init(); // implemented in biomeFuncs.hpp so constant memory can live there
                 // biomeFuncs.hpp included only by featurePlacement.hpp which is included only by chunk.cu

    std::vector<FeatureGen>& getBiomeFeatureGens(Biome biome);

    ivec2 getFeatureHeightBounds(Feature feature);
}