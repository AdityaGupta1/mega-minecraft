#pragma once

#include "block.hpp"

#define MAX_GATHERED_FEATURES_PER_CHUNK 1024

#define SEA_LEVEL 128

enum class Biome : unsigned char
{
    //CORAL_REEF,
    //ARCHIPELAGO,
    //WARM_OCEAN,
    COOL_OCEAN,

    //ROCKY_BEACH,
    //TROPICAL_BEACH,
    BEACH,

    SAVANNA,
    MESA,
    FROZEN_WASTELAND,
    REDWOOD_FOREST,
    SHREKS_SWAMP,
    SPARSE_DESERT,
    LUSH_BIRCH_FOREST,
    TIANZI_MOUNTAINS,

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
    Block grassBlock{ Block::DIRT };
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
    TERRACOTTA,
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
    SAND,
    SMOOTH_SAND,
    SNOW
};

static constexpr int numMaterials = (int)Material::SNOW + 1;
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

    ACACIA_TREE,

    REDWOOD_TREE,

    CYPRESS_TREE,

    BIRCH_TREE,

    PINE_TREE,
    PINE_SHRUB,

    RAFFLESIA,
    LARGE_JUNGLE_TREE,
    SMALL_JUNGLE_TREE,
    TINY_JUNGLE_TREE,

    // TINY_PURPLE_MUSHROOM
    // SMALL_PURPLE_MUSHROOM
    PURPLE_MUSHROOM,

    CRYSTAL,

    PALM_TREE,
    
    //JOSHUA_TREE,
    CACTUS
    
    //POND (not sure if this should go here or somewhere else)
};

static constexpr int numFeatures = (int)Feature::CACTUS + 1;

struct FeatureGenTopLayer
{
    Material material;
    float minThickness{ 0.0f };
};

struct FeatureGen
{
    Feature feature;
    int gridCellSize;
    int gridCellPadding;
    float chancePerGridCell;
    std::vector<FeatureGenTopLayer> possibleTopLayers;
    bool canReplaceBlocks{ true };
};

struct FeaturePlacement
{
    Feature feature;
    glm::ivec3 pos;
    bool canReplaceBlocks;
};

namespace BiomeUtils
{
    using namespace glm;

    void init(); // implemented in biomeFuncs.hpp so constant memory can live there
                 // biomeFuncs.hpp included only by featurePlacement.hpp which is included only by chunk.cu
}