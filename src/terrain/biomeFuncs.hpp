#pragma once

#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>
#include <unordered_map>

#pragma region utility functions

template<int xSize = 16>
__host__ __device__
int posTo2dIndex(const int x, const int z)
{
    return x + xSize * z;
}

template<int xSize = 16>
__host__ __device__
int posTo2dIndex(const ivec2 pos)
{
    return posTo2dIndex<xSize>(pos.x, pos.y);
}

template<int xSize = 16, int ySize = 384>
__host__ __device__
int posTo3dIndex(const int x, const int y, const int z)
{
    return y + ySize * posTo2dIndex<xSize>(x, z);
}

template<int xSize = 16, int ySize = 384>
__host__ __device__
int posTo3dIndex(const ivec3 pos)
{
    return posTo3dIndex<xSize, ySize>(pos.x, pos.y, pos.z);
}

__host__ __device__ Biome getRandomBiome(const float* columnBiomeWeights, float rand)
{
    for (int i = 0; i < numBiomes; ++i)
    {
        rand -= columnBiomeWeights[i];
        if (rand <= 0.f)
        {
            return (Biome)i;
        }
    }

    return Biome::PLAINS;
}

#pragma endregion

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
    case Biome::MOUNTAINS:
        return (1.f - moisture) * magic;
    }
}

__device__ float getHeight(Biome biome, vec2 pos)
{
    switch (biome)
    {
    case Biome::PLAINS:
        return 144.f + 8.f * fbm(pos * 0.0080f);
    case Biome::DESERT:
        return 134.f + 5.f * fbm(pos * 0.0100f);
    case Biome::PURPLE_MUSHROOMS:
        return 136.f + 9.f * fbm(pos * 0.0140f);
    case Biome::MOUNTAINS:
        float noise = pow(abs(fbm(pos * 0.0040f)) + 0.05f, 2.f);
        noise += ((fbm(pos * 0.0050f) - 0.5f) * 2.f) * 0.05f;
        return 165.f + 120.f * (noise - 0.2f);
    }
}

//__constant__ BiomeBlocks dev_biomeBlocks[numBiomes]; // TODO: convert to only top block for use with hashing transitions (replace generic top block with biome-specific top block)
__constant__ MaterialInfo dev_materialInfos[numMaterials];
__constant__ float dev_biomeMaterialWeights[numBiomes * numMaterials];

__constant__ ivec2 dev_dirVecs2d[8];

static std::array<std::vector<FeatureGen>, numBiomes> biomeFeatureGens;
static std::array<ivec2, numFeatures> featureHeightBounds;

void BiomeUtils::init()
{
    //BiomeBlocks* host_biomeBlocks = new BiomeBlocks[numBiomes];

    //host_biomeBlocks[(int)Biome::PLAINS] = { Block::GRASS, Block::DIRT, Block::STONE };
    //host_biomeBlocks[(int)Biome::DESERT] = { Block::SAND, Block::SAND, Block::STONE };
    //host_biomeBlocks[(int)Biome::PURPLE_MUSHROOMS] = { Block::MYCELIUM, Block::DIRT, Block::STONE };
    //host_biomeBlocks[(int)Biome::MOUNTAINS] = { Block::STONE, Block::STONE, Block::STONE };

    //cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, numBiomes * sizeof(BiomeBlocks));

    //delete[] host_biomeBlocks;

#pragma region material infos
    MaterialInfo* host_materialInfos = new MaterialInfo[numMaterials];

    // block, thickness, noise amplitude, noise scale
    host_materialInfos[(int)Material::BLACKSTONE] = { Block::BLACKSTONE, 56.f, 32.f, 0.0030f };
    host_materialInfos[(int)Material::DEEPSLATE] = { Block::DEEPSLATE, 52.f, 20.f, 0.0045f };
    host_materialInfos[(int)Material::SLATE] = { Block::SLATE, 6.f, 24.f, 0.0062f };
    host_materialInfos[(int)Material::STONE] = { Block::STONE, 32.f, 30.f, 0.0050f };
    host_materialInfos[(int)Material::TUFF] = { Block::TUFF, 24.f, 42.f, 0.0060f };
    host_materialInfos[(int)Material::CALCITE] = { Block::CALCITE, 20.f, 30.f, 0.0040f };
    host_materialInfos[(int)Material::GRANITE] = { Block::GRANITE, 18.f, 36.f, 0.0034f };
    host_materialInfos[(int)Material::MARBLE] = { Block::MARBLE, 28.f, 56.f, 0.0050f };
    host_materialInfos[(int)Material::ANDESITE] = { Block::ANDESITE, 24.f, 48.f, 0.0030f };

    // block, thickness, noise amplitude, noise scale
    //host_materialInfos[(int)Material::SANDSTONE] = { Block::SANDSTONE, 3.5f, 1.5f, 0.0025f };
    host_materialInfos[(int)Material::SANDSTONE] = { Block::SANDSTONE, 0.0f, 0.0f, 0.0025f };

    // block, thickness, angle of repose (degrees), maximum slope
    host_materialInfos[(int)Material::GRAVEL] = { Block::GRAVEL, 2.5f, 55.f, 1.8f };
    host_materialInfos[(int)Material::DIRT] = { Block::DIRT, 4.2f, 40.f, 1.2f };
    host_materialInfos[(int)Material::SAND] = { Block::SAND, 3.8f, 35.f, 1.4f };

    // convert angles of repose into their tangents
    for (int layerIdx = numStratifiedMaterials; layerIdx < numMaterials; ++layerIdx)
    {
        auto& materialInfo = host_materialInfos[layerIdx];
        materialInfo.noiseAmplitudeOrTanAngleOfRepose = tanf(glm::radians(materialInfo.noiseAmplitudeOrTanAngleOfRepose));
    }

    cudaMemcpyToSymbol(dev_materialInfos, host_materialInfos, numMaterials * sizeof(MaterialInfo));
    delete[] host_materialInfos;
#pragma endregion

#pragma region biome material weights
    float* host_biomeMaterialWeights = new float[numBiomes * numMaterials];

#define setCurrentBiomeMaterialWeight(material, weight) host_biomeMaterialWeights[posTo2dIndex<numMaterials>((int)Material::material, biomeIdx)] = weight
#define setBiomeMaterialWeight(biome, material, weight) host_biomeMaterialWeights[posTo2dIndex<numMaterials>((int)Material::material, (int)Biome::biome)] = weight

    for (int i = 0; i < numBiomes * numMaterials; ++i)
    {
        host_biomeMaterialWeights[i] = 1;
    }

    for (int biomeIdx = 0; biomeIdx < numBiomes; ++biomeIdx)
    {
        setCurrentBiomeMaterialWeight(SANDSTONE, 0);
        setCurrentBiomeMaterialWeight(GRAVEL, 0);
        setCurrentBiomeMaterialWeight(SAND, 0);
    }

    setBiomeMaterialWeight(MOUNTAINS, GRAVEL, 1);

    setBiomeMaterialWeight(DESERT, SANDSTONE, 1);
    setBiomeMaterialWeight(DESERT, DIRT, 0);
    setBiomeMaterialWeight(DESERT, SAND, 1);

#undef setCurrentBiomeMaterialWeight
#undef setBiomeMaterialWeight

    cudaMemcpyToSymbol(dev_biomeMaterialWeights, host_biomeMaterialWeights, numBiomes * numMaterials * sizeof(float));
    delete[] host_biomeMaterialWeights;
#pragma endregion

    cudaMemcpyToSymbol(dev_dirVecs2d, DirectionEnums::dirVecs2d.data(), 8 * sizeof(ivec2));

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