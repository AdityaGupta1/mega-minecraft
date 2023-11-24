#pragma once

#include "block.hpp"

#define MAX_CAVE_LAYERS_PER_COLUMN 32
#define MAX_GATHERED_FEATURES_PER_CHUNK 1024 // ~40 per chunk
#define MAX_GATHERED_CAVE_FEATURES_PER_CHUNK 1024 // ~40 per chunk

#define SEA_LEVEL 128
#define LAVA_LEVEL 8

enum class Biome : unsigned char
{
    CORAL_REEF,
    ARCHIPELAGO,
    WARM_OCEAN,
    ICEBERGS,
    COOL_OCEAN,

    ROCKY_BEACH,
    TROPICAL_BEACH,
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
static constexpr int numOceanBiomes = (int)Biome::COOL_OCEAN + 1;
static constexpr int numOceanAndBeachBiomes = (int)Biome::BEACH + 1;

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

struct CaveLayer
{
    CaveLayer() = default;

    int start; // exclusive
    int end; // inclusive
};

enum class Feature : unsigned char
{
    NONE,
    SPHERE,

    // CORAL (make multiple types of this)

    ICEBERG,

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

enum class CaveFeature : unsigned char
{
    NONE,
    TEST_PILLAR,

    STONE_PILLAR
};

static constexpr int numCaveFeatures = (int)CaveFeature::STONE_PILLAR + 1;

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

struct CaveFeaturePlacement
{
    CaveFeature feature;
    glm::ivec3 pos; // lowest air block of cave layer
    int height; // block at (pos.y + height) is highest air block of cave layer
    bool canReplaceBlocks;
};

namespace BiomeUtils
{
    using namespace glm;

    void init(); // implemented in biomeFuncs.hpp so constant memory can live there
                 // biomeFuncs.hpp included only by featurePlacement.hpp which is included only by chunk.cu
}