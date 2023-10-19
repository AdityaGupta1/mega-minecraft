#pragma once

#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>
#include <unordered_map>

__device__ float fbm(vec2 pos)
{
    float fbm = 0.f;
    float amplitude = 1.f;
    for (int i = 0; i < 4; ++i)
    {
        amplitude *= 0.5f;
        pos *= 2.f;
        fbm += amplitude * glm::simplex(pos);
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
    case Biome::MUSHROOMS:
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
        return 80.f + 12.f * fbm(pos * 0.01f);
    case Biome::DESERT:
        return 70.f + 5.f * fbm(pos * 0.005f);
    case Biome::MUSHROOMS:
        return 72.f + 6.f * fbm(pos * 0.004f);
    case Biome::METEORS:
        return 75.f + 7.f * fbm(pos * 0.007f);
    }
}

const std::unordered_map<Biome, std::vector<FeatureGen>> biomeFeatureGens = {
    {Biome::PLAINS, {{Feature::SPHERE, 0.0003f}}}
};

__constant__ BiomeBlocks dev_biomeBlocks[(int)Biome::numBiomes];

void BiomeUtils::init()
{
    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[(int)Biome::numBiomes];

    host_biomeBlocks[(int)Biome::PLAINS] = { Block::GRASS, Block::DIRT, Block::STONE };
    host_biomeBlocks[(int)Biome::DESERT] = { Block::SAND, Block::SAND, Block::STONE };
    host_biomeBlocks[(int)Biome::MUSHROOMS] = { Block::MYCELIUM, Block::DIRT, Block::STONE };
    host_biomeBlocks[(int)Biome::METEORS] = { Block::STONE, Block::STONE, Block::STONE };

    cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, (int)Biome::numBiomes * sizeof(BiomeBlocks));

    delete[] host_biomeBlocks;
}
