#pragma once

#include "biome.hpp"
#include "cuda/cudaUtils.hpp"
#include <unordered_map>
#include "defines.hpp"
#include "util/rng.hpp"

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

template<int stride = 1>
__host__ __device__
Biome getRandomBiome(const float* columnBiomeWeights, float rand)
{
    for (int i = 0; i < numBiomes; ++i)
    {
        rand -= columnBiomeWeights[stride * i];
        if (rand <= 0.f)
        {
            return (Biome)i;
        }
    }

    return Biome::PLAINS;
}

#pragma endregion

#pragma region biome weights

struct BiomeNoise
{
    float rocky;
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
    noise.rocky = getSingleBiomeNoise(biomeNoisePos, 0.0015f, vec2(-8102.35f, -7620.23f), 0.08f);
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
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.rocky, noise.rocky);
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.magic, noise.magic);
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.temperature, noise.temperature);
    applySingleBiomeNoise(totalWeight, biomeNoiseWeights.moisture, noise.moisture);
    return totalWeight;
}

#pragma endregion

__device__ float getHeight(Biome biome, vec2 pos)
{
    switch (biome)
    {
    case Biome::SAVANNA:
    {
        vec2 noiseOffsetPos = pos * 0.0040f;
        vec2 noiseOffset = fbm2<5>(noiseOffsetPos) * 100.f;

        vec2 noisePos = pos + noiseOffset;

        float plateauNoise1 = worley(noisePos * 0.0070f);
        plateauNoise1 = smoothstep(0.30f, 0.20f, plateauNoise1) * (1.f + 0.3f * simplex(noisePos * 0.0100f));

        float plateauNoise2 = worley((noisePos + vec2(-3910.12f, -9012.34f)) * 0.0045f);
        plateauNoise2 = smoothstep(0.16f, 0.08f, plateauNoise2) * (1.f + 0.2f * simplex(noisePos * 0.0130f));

        float plateauHeight = (plateauNoise1 * 14.f) + (plateauNoise2 * 9.f);
        return 136.f + 9.f * fbm<4>(pos * 0.0080f) + plateauHeight;
    }
    case Biome::MESA:
    {
        pos *= 0.7f;
        vec2 noiseOffsetPos = pos * 0.0050f;
        vec2 noiseOffset = fbm2<5>(noiseOffsetPos) * 300.f;
        float riverNoise = worleyEdgeDist((pos + noiseOffset) * 0.0030f);

        float baseHeight = 122.f;
        baseHeight += 10.f * smoothstep(0.00f, 0.05f, riverNoise);
        baseHeight += (37.5f + 5.0f * fbm<4>((pos + 0.02f * noiseOffset) * 0.0300f)) * smoothstep(0.07f, 0.22f, riverNoise);

        return baseHeight + 6.f * simplex(pos * 0.0250f);
    }
    case Biome::FROZEN_WASTELAND:
    {
        return 136.f + 16.f * fbm(pos * 0.0035f);
    }
    case Biome::REDWOOD_FOREST:
    {
        return 134.f + 8.f * fbm(pos * 0.0120f);
    }
    case Biome::SHREKS_SWAMP:
    {
        return 130.f + 12.f * fbm(pos * 0.0080f);
    }
    case Biome::SPARSE_DESERT:
    {
        vec2 noiseOffset = simplex2(pos * 0.0080f) * 20.0f;
        float dunesNoise = powf(worley((pos + noiseOffset) * 0.0160f), 2.f) * 18.f;
        return 132.f + 4.f * fbm<4>(pos * 0.0070f) + dunesNoise;
    }
    case Biome::LUSH_BIRCH_FOREST:
    {
        float hillsHeight = (simplex(pos * 0.0012f) + 0.8f) * 20.f;
        return 135.f + 8.f * fbm(pos * 0.0090f) + hillsHeight;
    }
    case Biome::TIANZI_MOUNTAINS:
    {
        vec2 noiseOffset = simplex2(pos * 0.0800f) * 3.0f;
        vec2 noisePos = (pos + noiseOffset) * 0.0300f;

        float worley1 = smoothstep(0.45f, 0.35f, worley(noisePos)) * 0.8f;
        float worley2 = smoothstep(0.45f, 0.35f, worley(noisePos * 1.8f)) * 0.2f;
        float mountainsHeight = worley1 + worley2;
        mountainsHeight *= 65.f + 5.f * fbm<3>(noisePos * 1.6f);

        float hillsHeight = 16.f * simplex(pos * 0.0150f);

        return 131.f + hillsHeight + 9.f * fbm<3>(pos * 0.0070f) + mountainsHeight;
    }
    case Biome::JUNGLE:
    {
        float hillsHeight = (simplex(pos * 0.0030f) + 0.5f) * 25.f;
        return 139.f + 8.f * fbm(pos * 0.0120f) + hillsHeight;
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
        float towersBaseNoise = simplex(pos * 0.0030f);

        vec3 worleyColor;
        float towersWorleyNoise = worleyEdgeDist(pos * 0.0700f, &worleyColor);
        towersWorleyNoise = smoothstep(0.10f, 0.15f, towersWorleyNoise);
        towersWorleyNoise *= 0.4f + 1.2f * worleyColor.r;
        float towersHeight = 60.f * towersWorleyNoise * smoothstep(0.70f, 0.74f, towersBaseNoise);
        towersHeight += 18.f * smoothstep(0.35f, 0.8f, towersBaseNoise);

        float baseHeight = 137.f + 8.f * fbm(pos * 0.0200f);
        return baseHeight + towersHeight;
    }
    case Biome::OASIS:
    {
        return 132.f + 9.f * fbm(pos * 0.0120f);
    }
    case Biome::DESERT:
    {
        return 136.f + 6.f * fbm(pos * 0.0110f);
    }
    case Biome::PLAINS:
    {
        return 144.f + 8.f * fbm(pos * 0.0080f);
    }
    case Biome::MOUNTAINS:
    {
        float noise = powf(abs(fbm(pos * 0.0035f)) + 0.05f, 2.f);
        noise += ((fbm(pos * 0.0050f) - 0.5f) * 2.f) * 0.05f;
        return 165.f + (140.f * (noise - 0.15f)) + (noise * (20.f * fbm(pos * 0.0350f)));
    }
    }

    printf("getHeight() reached an unreachable section");
    return SEA_LEVEL;
}

__device__ bool biomeBlockPreProcess(Block* block, Biome biome, vec3 worldBlockPos, float height)
{
    switch (biome)
    {
    case Biome::CRYSTALS:
    {
        if (height > 176.f)
        {
            float quartzStartHeight = 140.f + 15.f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0080f);
            if (worldBlockPos.y > quartzStartHeight)
            {
                *block = Block::QUARTZ;
                return true;
            }
        }

        return false;
    }
    }

    return false;
}

__device__ bool biomeBlockPostProcess(Block* block, Biome biome, vec3 worldBlockPos, float height, bool isTopBlock)
{
    switch (biome)
    {
    case Biome::MESA:
    {
        if (worldBlockPos.y < 90.f || *block == Block::WATER)
        {
            return false;
        }

        vec2 pos2d = vec2(worldBlockPos.x, worldBlockPos.z);
        float terracottaStartHeight = 108.f + 12.f * fbm<3>(pos2d * 0.0040f);
        if (worldBlockPos.y < terracottaStartHeight)
        {
            return false;
        }

        if (*block == Block::CLAY && worldBlockPos.y < terracottaStartHeight + 20.f)
        {
            return false;
        }

        float sampleHeight = worldBlockPos.y + 3.f * simplex(vec3(pos2d * 0.0100f, worldBlockPos.y * 0.0300f)) - terracottaStartHeight;
        sampleHeight = mod(sampleHeight, 32.f);
        Block terracottaBlock;
        if (sampleHeight < 5.f)
        {
            terracottaBlock = Block::TERRACOTTA;
        }
        else if (sampleHeight < 8.f)
        {
            terracottaBlock = Block::ORANGE_TERRACOTTA;
        }
        else if (sampleHeight < 12.f)
        {
            terracottaBlock = Block::RED_TERRACOTTA;
        }
        else if (sampleHeight < 14.f)
        {
            terracottaBlock = Block::WHITE_TERRACOTTA;
        }
        else if (sampleHeight < 20.f)
        {
            terracottaBlock = Block::TERRACOTTA;
        }
        else if (sampleHeight < 21.f)
        {
            terracottaBlock = Block::ORANGE_TERRACOTTA;
        }
        else if (sampleHeight < 26.f)
        {
            terracottaBlock = Block::YELLOW_TERRACOTTA;
        }
        else if (sampleHeight < 29.f)
        {
            terracottaBlock = Block::PURPLE_TERRACOTTA;
        }
        else
        {
            terracottaBlock = Block::TERRACOTTA;
        }

        *block = terracottaBlock;
        return true;
    }
    case Biome::FROZEN_WASTELAND:
    {
        if (*block != Block::WATER)
        {
            return false;
        }

        *block = Block::PACKED_ICE;
        return true;
    }
    case Biome::SHREKS_SWAMP:
    {
        if (worldBlockPos.y < 100.f)
        {
            return false;
        }

        if (*block == Block::DIRT || *block == Block::JUNGLE_GRASS)
        {
            float mudEnd = SEA_LEVEL + 0.8f + 1.1f * simplex(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0300f);
            if (worldBlockPos.y < mudEnd)
            {
                *block = Block::MUD;
                return true;
            }
        }

        return false;
    }
    case Biome::TIANZI_MOUNTAINS:
    {
        if (worldBlockPos.y < 90.f || *block == Block::WATER || *block == Block::DIRT || *block == Block::GRASS)
        {
            return false;
        }

        float sandstoneStartHeight = 112.f + 14.f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0040f);

        if (worldBlockPos.y < sandstoneStartHeight)
        {
            return false;
        }

        *block = Block::SMOOTH_SANDSTONE;
        return true;
    }
    case Biome::MOUNTAINS:
    {
        //if (!isTopBlock)
        //{
        //    return false;
        //}
        
        if (worldBlockPos.y < 190.f)
        {
            return false;
        }

        float snowStartHeight = 202.f + 5.f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0500f);
        if (worldBlockPos.y < snowStartHeight)
        {
            return false;
        }

        *block = Block::SNOW;
        return true;
    }
    }

    return false;
}

__constant__ BiomeBlocks dev_biomeBlocks[numBiomes];
__constant__ MaterialInfo dev_materialInfos[numMaterials];
__constant__ float dev_biomeMaterialWeights[numBiomes * numMaterials];

__constant__ ivec2 dev_dirVecs2d[8];

static std::array<std::vector<FeatureGen>, numBiomes> host_biomeFeatureGens;
static std::array<ivec2, numFeatures> host_featureHeightBounds;
__constant__ ivec2 dev_featureHeightBounds[numFeatures];

void BiomeUtils::init()
{
    BiomeNoise* host_biomeNoiseWeights = new BiomeNoise[numBiomes];

    host_biomeNoiseWeights[(int)Biome::SAVANNA] = { 1, 1, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::MESA] = { 1, 1, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::FROZEN_WASTELAND] = { 1, 1, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::REDWOOD_FOREST] = { 1, 1, 0, 0 };
    host_biomeNoiseWeights[(int)Biome::SHREKS_SWAMP] = { 1, 0, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::SPARSE_DESERT] = { 1, 0, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::LUSH_BIRCH_FOREST] = { 1, 0, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::TIANZI_MOUNTAINS] = { 1, 0, 0, 0 };

    host_biomeNoiseWeights[(int)Biome::JUNGLE] = { 0, 1, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::RED_DESERT] = { 0, 1, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::PURPLE_MUSHROOMS] = { 0, 1, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::CRYSTALS] = { 0, 1, 0, 0 };
    host_biomeNoiseWeights[(int)Biome::OASIS] = { 0, 0, 1, 1 };
    host_biomeNoiseWeights[(int)Biome::DESERT] = { 0, 0, 1, 0 };
    host_biomeNoiseWeights[(int)Biome::PLAINS] = { 0, 0, 0, 1 };
    host_biomeNoiseWeights[(int)Biome::MOUNTAINS] = { 0, 0, 0, 0 };

    cudaMemcpyToSymbol(dev_biomeNoiseWeights, host_biomeNoiseWeights, numBiomes * sizeof(BiomeNoise));
    delete[] host_biomeNoiseWeights;

    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[numBiomes];

    host_biomeBlocks[(int)Biome::SAVANNA].grassBlock = Block::SAVANNA_GRASS; // TODO: replace with regular grass after implementing biome-based color
    host_biomeBlocks[(int)Biome::FROZEN_WASTELAND].grassBlock = Block::SNOWY_GRASS;
    host_biomeBlocks[(int)Biome::REDWOOD_FOREST].grassBlock = Block::GRASS;
    host_biomeBlocks[(int)Biome::SHREKS_SWAMP].grassBlock = Block::JUNGLE_GRASS;
    host_biomeBlocks[(int)Biome::LUSH_BIRCH_FOREST].grassBlock = Block::GRASS;
    host_biomeBlocks[(int)Biome::TIANZI_MOUNTAINS].grassBlock = Block::GRASS;

    host_biomeBlocks[(int)Biome::JUNGLE].grassBlock = Block::JUNGLE_GRASS; // TODO: replace with regular grass after implementing biome-based color
    host_biomeBlocks[(int)Biome::PURPLE_MUSHROOMS].grassBlock = Block::MYCELIUM;
    host_biomeBlocks[(int)Biome::OASIS].grassBlock = Block::JUNGLE_GRASS;
    host_biomeBlocks[(int)Biome::PLAINS].grassBlock = Block::GRASS;
    host_biomeBlocks[(int)Biome::MOUNTAINS].grassBlock = Block::GRASS;

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
    setMaterialInfoSameBlock(TERRACOTTA, 32.f, 16.f, 0.0020f);
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
    setMaterialInfoSameBlock(SMOOTH_SAND, 4.5f, 65.f, 4.0f);
    setMaterialInfoSameBlock(SNOW, 2.5f, 45.f, 1.5f);

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
        setCurrentBiomeMaterialWeight(TERRACOTTA, 0.0f);

        setCurrentBiomeMaterialWeight(RED_SANDSTONE, 0.0f);
        setCurrentBiomeMaterialWeight(SANDSTONE, 0.0f);

        setCurrentBiomeMaterialWeight(GRAVEL, 0.0f);
        setCurrentBiomeMaterialWeight(CLAY, 0.0f);
        setCurrentBiomeMaterialWeight(MUD, 0.0f);
        setCurrentBiomeMaterialWeight(RED_SAND, 0.0f);
        setCurrentBiomeMaterialWeight(SAND, 0.0f);
        setCurrentBiomeMaterialWeight(SMOOTH_SAND, 0.0f);
        setCurrentBiomeMaterialWeight(SNOW, 0.0f);
    }

    setBiomeMaterialWeight(SAVANNA, STONE, 0.6f);
    setBiomeMaterialWeight(SAVANNA, TUFF, 0.15f);
    setBiomeMaterialWeight(SAVANNA, CALCITE, 0.0f);
    setBiomeMaterialWeight(SAVANNA, GRANITE, 0.2f);
    setBiomeMaterialWeight(SAVANNA, TERRACOTTA, 3.2f);
    setBiomeMaterialWeight(SAVANNA, MARBLE, 0.0f);

    setBiomeMaterialWeight(MESA, CLAY, 0.8f);
    setBiomeMaterialWeight(MESA, DIRT, 0.0f);

    setBiomeMaterialWeight(FROZEN_WASTELAND, GRANITE, 0.0f);
    setBiomeMaterialWeight(FROZEN_WASTELAND, DIRT, 0.6f);
    setBiomeMaterialWeight(FROZEN_WASTELAND, SNOW, 1.1f);

    setBiomeMaterialWeight(SHREKS_SWAMP, CLAY, 1.7f);
    setBiomeMaterialWeight(SHREKS_SWAMP, MUD, 2.2f);
    setBiomeMaterialWeight(SHREKS_SWAMP, DIRT, 0.6f);

    setBiomeMaterialWeight(SPARSE_DESERT, MARBLE, 2.0f);
    setBiomeMaterialWeight(SPARSE_DESERT, ANDESITE, 0.5f);
    setBiomeMaterialWeight(SPARSE_DESERT, DIRT, 0.0f);
    setBiomeMaterialWeight(SPARSE_DESERT, SMOOTH_SAND, 1.4f);

    setBiomeMaterialWeight(TIANZI_MOUNTAINS, SANDSTONE, 1.0f);

    setBiomeMaterialWeight(JUNGLE, CLAY, 1.0f);
    setBiomeMaterialWeight(JUNGLE, MUD, 1.0f);
    setBiomeMaterialWeight(JUNGLE, DIRT, 0.5f);

    setBiomeMaterialWeight(RED_DESERT, RED_SANDSTONE, 1.0f);
    setBiomeMaterialWeight(RED_DESERT, DIRT, 0.0f);
    setBiomeMaterialWeight(RED_DESERT, RED_SAND, 1.0f);

    setBiomeMaterialWeight(PURPLE_MUSHROOMS, GRAVEL, 0.4f);

    setBiomeMaterialWeight(CRYSTALS, CALCITE, 0.3f);
    setBiomeMaterialWeight(CRYSTALS, GRAVEL, 0.15f);
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

    // feature, gridCellSize, gridCellPadding, chancePerGridCell, possibleTopLayers
    host_biomeFeatureGens[(int)Biome::SAVANNA] = {
        { Feature::ACACIA_TREE, 36, 4, 0.3f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::REDWOOD_FOREST] = {
        { Feature::REDWOOD_TREE, 10, 2, 0.75f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::SHREKS_SWAMP] = {
        { Feature::CYPRESS_TREE, 18, 3, 0.6f, {} },
        { Feature::BIRCH_TREE, 16, 2, 0.15f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::LUSH_BIRCH_FOREST] = {
        { Feature::BIRCH_TREE, 9, 2, 0.7f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::TIANZI_MOUNTAINS] = {
        { Feature::PINE_TREE, 7, 1, 0.80f, {}, false },
        { Feature::PINE_SHRUB, 6, 1, 0.80f, {}, false }
    };

    host_biomeFeatureGens[(int)Biome::JUNGLE] = {
        { Feature::RAFFLESIA, 54, 6, 0.50f, { {Material::DIRT, 0.5f} } },
        { Feature::LARGE_JUNGLE_TREE, 32, 3, 0.70f, { {Material::DIRT, 0.5f} } },
        { Feature::SMALL_JUNGLE_TREE, 10, 2, 0.75f, { {Material::DIRT, 0.5f} } },
        { Feature::TINY_JUNGLE_TREE, 6, 1, 0.18f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::RED_DESERT] = {
        { Feature::PALM_TREE, 40, 3, 0.20f, { {Material::RED_SAND, 0.3f} } },
        { Feature::CACTUS, 16, 2, 0.20f, { {Material::RED_SAND, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::PURPLE_MUSHROOMS] = {
        { Feature::PURPLE_MUSHROOM, 11, 3, 0.45f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::CRYSTALS] = {
        { Feature::CRYSTAL, 56, 12, 0.8f, { {Material::STONE, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::OASIS] = {
        { Feature::PALM_TREE, 24, 3, 0.35f, { {Material::SAND, 0.3f} } },
        { Feature::CACTUS, 16, 2, 0.40f, { {Material::SAND, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::DESERT] = {
        { Feature::PALM_TREE, 64, 3, 0.30f, { {Material::SAND, 0.3f} } },
        { Feature::CACTUS, 16, 2, 0.70f, { {Material::SAND, 0.5f} } }
    };

#define setFeatureHeightBounds(feature, yMin, yMax) host_featureHeightBounds[(int)Feature::feature] = ivec2(yMin, yMax)

    setFeatureHeightBounds(NONE, 0, 0);
    setFeatureHeightBounds(SPHERE, -6, 6);

    setFeatureHeightBounds(ACACIA_TREE, 0, 15);

    setFeatureHeightBounds(REDWOOD_TREE, -5, 75);

    setFeatureHeightBounds(CYPRESS_TREE, -3, 65); // TODO: revisit

    setFeatureHeightBounds(BIRCH_TREE, 0, 30);

    setFeatureHeightBounds(PINE_TREE, 0, 15);
    setFeatureHeightBounds(PINE_SHRUB, 0, 8);

    setFeatureHeightBounds(RAFFLESIA, 0, 10);
    setFeatureHeightBounds(TINY_JUNGLE_TREE, 0, 5);
    setFeatureHeightBounds(SMALL_JUNGLE_TREE, 0, 17);
    setFeatureHeightBounds(LARGE_JUNGLE_TREE, 0, 38);

    setFeatureHeightBounds(PURPLE_MUSHROOM, 0, 80);

    setFeatureHeightBounds(CRYSTAL, -4, 65);

    setFeatureHeightBounds(PALM_TREE, 0, 28);

    setFeatureHeightBounds(CACTUS, 0, 15);

#undef setFeatureHeightBounds

    cudaMemcpyToSymbol(dev_featureHeightBounds, host_featureHeightBounds.data(), numFeatures * sizeof(ivec2));
}