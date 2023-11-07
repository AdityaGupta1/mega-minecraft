#pragma once

#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <glm/gtc/noise.hpp>
#include <unordered_map>
#include "defines.hpp"

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

struct BiomeNoise
{
    float magic;
    float temperature;
    float moisture;
};

static constexpr float overallBiomeScale = 0.32f;
__constant__ BiomeNoise dev_biomeNoiseWeights[numBiomes];

__device__ float getSingleBiomeNoise(vec2 pos, float noiseScale, vec2 offset, float smoothstepThreshold)
{
    return glm::smoothstep(-smoothstepThreshold, smoothstepThreshold, glm::simplex(pos * noiseScale + offset));
}

__device__ BiomeNoise getBiomeNoise(const vec2 worldPos)
{
    const vec2 noiseOffset = vec2(
        glm::simplex(worldPos * 0.015f + vec2(6839.19f, 1803.34f)),
        glm::simplex(worldPos * 0.015f + vec2(8230.53f, 2042.84f))
    ) * 14.f;
    const vec2 biomeNoisePos = (worldPos + noiseOffset) * overallBiomeScale;

    BiomeNoise noise;
    noise.magic = getSingleBiomeNoise(biomeNoisePos, 0.0030f, vec2(5612.35f, 9182.49f), 0.07f);
    noise.temperature = getSingleBiomeNoise(biomeNoisePos, 0.0012f, vec2(-4021.34f, -8720.12f), 0.06f);
    noise.moisture = getSingleBiomeNoise(biomeNoisePos, 0.0050f, vec2(1835.32f, 3019.39f), 0.12f);
    return noise;
}

__device__ void applySingleBiomeNoise(float& totalWeight, const float noise, const float weight)
{
    if (weight >= 0)
    {
        totalWeight *= glm::mix(1.f - noise, noise, weight);
    }
}

__device__ float getBiomeWeight(Biome biome, const BiomeNoise& noise)
{
    const auto& biomeNoiseWeights = dev_biomeNoiseWeights[(int)biome];

    float totalWeight = 1.f;
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.magic, noise.magic);
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.temperature, noise.temperature);
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.moisture, noise.moisture);
    return totalWeight;
}

__device__ float getHeight(Biome biome, vec2 pos)
{
    switch (biome)
    {
    case Biome::JUNGLE:
    {
        float hillsNoise = (glm::simplex(pos * 0.0030f) + 0.5f) * 25.f;
        return 139.f + 8.f * fbm(pos * 0.0120f) + hillsNoise;
    }
    case Biome::RED_DESERT:
    {
        return 137.f + 13.f * fbm(pos * 0.0075f);
    }
    case Biome::PURPLE_MUSHROOMS:
    {
        return 136.f + 9.f * fbm(pos * 0.0140f);
    }
    case Biome::CRYSTALS:
    {
        return 129.f + 7.f * fbm(pos * 0.0200f);
    }
    case Biome::OASIS:
    {
        return 134.f + 9.f * fbm(pos * 0.0120f);
    }
    case Biome::DESERT:
    {
        return 134.f + 6.f * fbm(pos * 0.0110f);
    }
    case Biome::PLAINS:
    {
        return 144.f + 8.f * fbm(pos * 0.0080f);
    }
    case Biome::MOUNTAINS:
    {
        float noise = pow(abs(fbm(pos * 0.0035f)) + 0.05f, 2.f);
        noise += ((fbm(pos * 0.0050f) - 0.5f) * 2.f) * 0.05f;
        return 165.f + (140.f * (noise - 0.15f)) + (noise * (20.f * fbm(pos * 0.0350f)));
    }
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
    BiomeNoise* host_biomeNoiseWeights = new BiomeNoise[numBiomes];

    host_biomeNoiseWeights[(int)Biome::JUNGLE] = { 1, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::RED_DESERT] = { 1, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::PURPLE_MUSHROOMS] = { 1, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::CRYSTALS] = { 1, 0, 0 };
    host_biomeNoiseWeights[(int)Biome::OASIS] = { 0, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::DESERT] = { 0, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::PLAINS] = { 0, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::MOUNTAINS] = { 0, 0, 0 };

    cudaMemcpyToSymbol(dev_biomeNoiseWeights, host_biomeNoiseWeights, numBiomes * sizeof(BiomeNoise));
    delete[] host_biomeNoiseWeights;

    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[numBiomes];

    host_biomeBlocks[(int)Biome::JUNGLE].grassBlock = Block::JUNGLE_GRASS;
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
    setMaterialInfoSameBlock(RED_SANDSTONE, 3.0f, 2.0f, 0.0035f);
    setMaterialInfoSameBlock(SANDSTONE, 3.5f, 1.5f, 0.0025f);

    // material/block, thickness, angle of repose (degrees), maximum slope
    setMaterialInfoSameBlock(GRAVEL, 2.5f, 55.f, 1.8f);
    setMaterialInfoSameBlock(CLAY, 2.7f, 40.f, 1.8f);
    setMaterialInfoSameBlock(MUD, 2.3f, 45.f, 1.6f);
    setMaterialInfoSameBlock(DIRT, 4.2f, 40.f, 1.2f);
    setMaterialInfoSameBlock(RED_SAND, 3.5f, 30.f, 1.5f);
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
        setCurrentBiomeMaterialWeight(RED_SANDSTONE, 0);
        setCurrentBiomeMaterialWeight(SANDSTONE, 0);

        setCurrentBiomeMaterialWeight(GRAVEL, 0);
        setCurrentBiomeMaterialWeight(CLAY, 0);
        setCurrentBiomeMaterialWeight(MUD, 0);
        setCurrentBiomeMaterialWeight(RED_SAND, 0);
        setCurrentBiomeMaterialWeight(SAND, 0);
    }

    setBiomeMaterialWeight(JUNGLE, CLAY, 1.0f);
    setBiomeMaterialWeight(JUNGLE, MUD, 1.0f);
    setBiomeMaterialWeight(JUNGLE, DIRT, 0.5f);

    setBiomeMaterialWeight(RED_DESERT, RED_SANDSTONE, 1.0f);
    setBiomeMaterialWeight(RED_DESERT, DIRT, 0.0f);
    setBiomeMaterialWeight(RED_DESERT, RED_SAND, 1.0f);

    setBiomeMaterialWeight(PURPLE_MUSHROOMS, GRAVEL, 0.4f);

    setBiomeMaterialWeight(CRYSTALS, GRAVEL, 0.35f);
    setBiomeMaterialWeight(CRYSTALS, CLAY, 0.2f);
    setBiomeMaterialWeight(CRYSTALS, DIRT, 0.0f);

    setBiomeMaterialWeight(OASIS, SANDSTONE, 1.0f);
    setBiomeMaterialWeight(OASIS, CLAY, 0.4f);
    setBiomeMaterialWeight(OASIS, DIRT, 0.6f);
    setBiomeMaterialWeight(OASIS, SAND, 0.4f);

    setBiomeMaterialWeight(DESERT, SANDSTONE, 1.0f);
    setBiomeMaterialWeight(DESERT, DIRT, 0.0f);
    setBiomeMaterialWeight(DESERT, SAND, 1.0f);

    setBiomeMaterialWeight(MOUNTAINS, GRAVEL, 1.0f);

#undef setCurrentBiomeMaterialWeight
#undef setBiomeMaterialWeight

    cudaMemcpyToSymbol(dev_biomeMaterialWeights, host_biomeMaterialWeights, numBiomes * numMaterials * sizeof(float));
    delete[] host_biomeMaterialWeights;
#pragma endregion

    cudaMemcpyToSymbol(dev_dirVecs2d, DirectionEnums::dirVecs2d.data(), 8 * sizeof(ivec2));

    // feature, chancePerBlock, possibleTopLayers
    biomeFeatureGens[(int)Biome::JUNGLE] = { 
        { Feature::RAFFLESIA, 0.0002f, { {Material::DIRT, 0.5f} } },
        { Feature::SMALL_JUNGLE_TREE, 0.0020f, { {Material::DIRT, 0.5f} } }
    };

    biomeFeatureGens[(int)Biome::PURPLE_MUSHROOMS] = {
        { Feature::PURPLE_MUSHROOM, 0.0020f, { {Material::DIRT, 0.5f} } }
    };

#define setFeatureHeightBounds(feature, yMin, yMax) featureHeightBounds[(int)Feature::feature] = ivec2(yMin, yMax)

    setFeatureHeightBounds(NONE, 0, 0);
    setFeatureHeightBounds(SPHERE, -6, 6);
    setFeatureHeightBounds(PURPLE_MUSHROOM, -2, 80);
    setFeatureHeightBounds(RAFFLESIA, -2, 10);
    setFeatureHeightBounds(SMALL_JUNGLE_TREE, -2, 17);

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