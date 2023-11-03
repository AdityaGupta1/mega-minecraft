#pragma once

#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>
#include <unordered_map>

__device__ float fbm(vec2 pos)
{
    float fbm = 0.f;
    float amplitude = 1.f;
    for (int i = 0; i < 5; ++i)
    {
        amplitude *= 0.5f;
        fbm += amplitude * glm::simplex(pos);
        pos *= 2.f;
    }
    return fbm;
}

__device__ float getBiomeWeight(Biome biome, float moisture, float magic)
{
    switch (biome)
    {
    case Biome::PLAINS:
        return moisture * (1.f - magic);
    case Biome::DESERT:
        return (1.f - moisture) * (1.f - magic);
    case Biome::PURPLE_MUSHROOMS:
        return moisture * magic;
    case Biome::METEORS:
        return (1.f - moisture) * magic;
    }
}

__device__ float getHeight(Biome biome, vec2 pos)
{
    switch (biome)
    {
    case Biome::PLAINS:
        return 144.f + 8.f * fbm(pos * 0.016f);
    case Biome::DESERT:
        return 134.f + 5.f * fbm(pos * 0.010f);
    case Biome::PURPLE_MUSHROOMS:
        return 136.f + 6.f * fbm(pos * 0.008f);
    case Biome::METEORS:
        //float simplex = pow(abs(fbm(pos * 0.0027f)) + 0.05f, 2.f) * 4.f;
        //return 139.f + 60.f * simplex;
        float noise = pow(abs(fbm(pos * 0.0015f)) + 0.05f, 2.f);
        noise += ((fbm(pos * 0.0050f) - 0.5f) * 2.f) * 0.05f;
        return 196.f + 180.f * (noise - 0.5f);
    }
}

//__constant__ BiomeBlocks dev_biomeBlocks[(int)Biome::numBiomes]; // TODO: convert to only top block for use with hashing transitions (replace generic top block with biome-specific top block)
__constant__ MaterialInfo dev_materialInfos[(int)Material::numMaterials];

static std::array<std::vector<FeatureGen>, (int)Biome::numBiomes> biomeFeatureGens;
static std::array<ivec2, (int)Feature::numFeatures> featureHeightBounds;

void BiomeUtils::init()
{
    //BiomeBlocks* host_biomeBlocks = new BiomeBlocks[(int)Biome::numBiomes];

    //host_biomeBlocks[(int)Biome::PLAINS] = { Block::GRASS, Block::DIRT, Block::STONE };
    //host_biomeBlocks[(int)Biome::DESERT] = { Block::SAND, Block::SAND, Block::STONE };
    //host_biomeBlocks[(int)Biome::PURPLE_MUSHROOMS] = { Block::MYCELIUM, Block::DIRT, Block::STONE };
    //host_biomeBlocks[(int)Biome::METEORS] = { Block::STONE, Block::STONE, Block::STONE };

    //cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, (int)Biome::numBiomes * sizeof(BiomeBlocks));

    //delete[] host_biomeBlocks;

    MaterialInfo* host_materialInfos = new MaterialInfo[(int)Material::numMaterials];

    host_materialInfos[(int)Material::BLACKSTONE] = { Block::BLACKSTONE, 56.f, 32.f, 0.0030f };
    host_materialInfos[(int)Material::DEEPSLATE] = { Block::DEEPSLATE, 48.f, 24.f, 0.0045f };
    host_materialInfos[(int)Material::SLATE] = { Block::SLATE, 4.f, 24.f, 0.0062f };
    host_materialInfos[(int)Material::STONE] = { Block::STONE, 36.f, 30.f, 0.0050f };
    host_materialInfos[(int)Material::TUFF] = { Block::TUFF, 32.f, 42.f, 0.0060f };
    host_materialInfos[(int)Material::CALCITE] = { Block::CALCITE, 20.f, 30.f, 0.0040f };
    host_materialInfos[(int)Material::GRANITE] = { Block::GRANITE, 18.f, 36.f, 0.0034f };
    host_materialInfos[(int)Material::MARBLE] = { Block::MARBLE, 24.f, 56.f, 0.0050f };
    host_materialInfos[(int)Material::ANDESITE] = { Block::ANDESITE, 24.f, 48.f, 0.0030f };

    host_materialInfos[(int)Material::DIRT] = { Block::DIRT };

    cudaMemcpyToSymbol(dev_materialInfos, host_materialInfos, (int)Material::numMaterials * sizeof(MaterialInfo));

    delete[] host_materialInfos;

    biomeFeatureGens[(int)Biome::PURPLE_MUSHROOMS] = { {Feature::PURPLE_MUSHROOM, 0.004f} };

    featureHeightBounds[(int)Feature::NONE] = ivec2(0, 0);
    featureHeightBounds[(int)Feature::SPHERE] = ivec2(-6, -6);
    featureHeightBounds[(int)Feature::PURPLE_MUSHROOM] = ivec2(-2, 80);
}

std::vector<FeatureGen>& BiomeUtils::getBiomeFeatureGens(Biome biome)
{
    return biomeFeatureGens[(int)biome];
}

ivec2 BiomeUtils::getFeatureHeightBounds(Feature feature)
{
    return featureHeightBounds[(int)feature];
}