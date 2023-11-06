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

__constant__ BiomeBlocks dev_biomeBlocks[numBiomes];
__constant__ MaterialInfo dev_materialInfos[numMaterials];
__constant__ float dev_biomeMaterialWeights[numBiomes * numMaterials];

__constant__ ivec2 dev_dirVecs2d[8];

static std::array<std::vector<FeatureGen>, numBiomes> biomeFeatureGens;
static std::array<ivec2, numFeatures> featureHeightBounds;

void BiomeUtils::init()
{
    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[numBiomes];

    host_biomeBlocks[(int)Biome::PURPLE_MUSHROOMS].grassBlock = Block::MYCELIUM;

    cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, numBiomes * sizeof(BiomeBlocks));
    delete[] host_biomeBlocks;

#pragma region material infos
    MaterialInfo* host_materialInfos = new MaterialInfo[numMaterials];

#define setMaterialInfo(material, block, v1, v2, v3) host_materialInfos[(int)Material::material] = { Block::block, v1, v2, v3 }
#define setMaterialInfoSameBlock(material, v1, v2, v3) setMaterialInfo(material, material, v1, v2, v3)

    // material/block, thickness, noise amplitude, noise scale
    setMaterialInfoSameBlock(BLACKSTONE, 56.f, 32.f, 0.0030f);
    setMaterialInfoSameBlock(DEEPSLATE, 52.f, 20.f, 0.0045f);
    setMaterialInfoSameBlock(SLATE, 6.f, 24.f, 0.0062f);
    setMaterialInfoSameBlock(STONE, 32.f, 30.f, 0.0050f);
    setMaterialInfoSameBlock(TUFF, 24.f, 42.f, 0.0060f);
    setMaterialInfoSameBlock(CALCITE, 20.f, 30.f, 0.0040f);
    setMaterialInfoSameBlock(GRANITE, 18.f, 36.f, 0.0034f);
    setMaterialInfoSameBlock(MARBLE, 28.f, 56.f, 0.0050f);
    setMaterialInfoSameBlock(ANDESITE, 24.f, 48.f, 0.0030f);

    // material/block, thickness, noise amplitude, noise scale
    setMaterialInfoSameBlock(SANDSTONE, 3.5f, 1.5f, 0.0025f);

    // material/block, thickness, angle of repose (degrees), maximum slope
    setMaterialInfoSameBlock(GRAVEL, 2.5f, 55.f, 1.8f);
    setMaterialInfoSameBlock(DIRT, 4.2f, 40.f, 1.2f);
    setMaterialInfoSameBlock(SAND, 3.8f, 35.f, 1.4f);

#undef setMaterialInfo
#undef setMaterialInfoSameBlock

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

#define setFeatureHeightBounds(feature, yMin, yMax) featureHeightBounds[(int)Feature::feature] = ivec2(yMin, yMax)

    setFeatureHeightBounds(NONE, 0, 0);
    setFeatureHeightBounds(SPHERE, -6, -6);
    setFeatureHeightBounds(PURPLE_MUSHROOM, -2, 80);

#undef setFeatureHeightBounds
}

std::vector<FeatureGen>& BiomeUtils::getBiomeFeatureGens(Biome biome)
{
    return biomeFeatureGens[(int)biome];
}

ivec2 BiomeUtils::getFeatureHeightBounds(Feature feature)
{
    return featureHeightBounds[(int)feature];
}