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
    float ocean;
    float beach;
    float rocky;
    float magic;
    float temperature;
    float moisture;
};

struct CaveBiomeNoise
{
    float none;
    float shallow;

    float warped;
    float rocky;
};

enum class BiomeWeightType : unsigned char
{
    W_IGNORE,
    W_POSITIVE,
    W_NEGATIVE
};

struct BiomeWeights
{
    BiomeWeightType ocean;
    BiomeWeightType beach;
    BiomeWeightType rocky;
    BiomeWeightType magic;
    BiomeWeightType temperature;
    BiomeWeightType moisture;
};

struct CaveBiomeWeights
{
    BiomeWeightType none;
    BiomeWeightType shallow;

    BiomeWeightType warped;
    BiomeWeightType rocky;
};

static constexpr float overallBiomeScale = 0.32f;
__constant__ BiomeWeights dev_biomeNoiseWeights[numBiomes];
static constexpr float overallCaveBiomeScale = 1.f;
__constant__ CaveBiomeWeights dev_caveBiomeNoiseWeights[numCaveBiomes];

__device__ float getSingleBiomeNoise(vec2 pos, float noiseScale, vec2 offset, float smoothstepThreshold)
{
    return smoothstep(-smoothstepThreshold, smoothstepThreshold, simplex(pos * noiseScale + offset));
}

__device__ BiomeNoise getBiomeNoise(const vec2 worldBlockPos)
{
    const vec2 noiseOffset = fbm2From2<3>(worldBlockPos * 0.0150f) * 20.f;
    const vec2 biomeNoisePos = (worldBlockPos + noiseOffset) * overallBiomeScale;

    BiomeNoise noise;
    float oceanNoise = simplex(biomeNoisePos * 0.0007f + vec2(2853.49f, -9481.42f));
    noise.ocean = smoothstep(0.01f, -0.02f, oceanNoise);
    noise.beach = smoothstep(-0.15f, -0.05f, oceanNoise);
    noise.rocky = getSingleBiomeNoise(biomeNoisePos, 0.0015f, vec2(-8102.35f, -7620.23f), 0.08f);
    noise.magic = getSingleBiomeNoise(biomeNoisePos, 0.0030f, vec2(5612.35f, 9182.49f), 0.07f);
    noise.temperature = getSingleBiomeNoise(biomeNoisePos, 0.0012f, vec2(-4021.34f, -8720.12f), 0.06f);
    noise.moisture = getSingleBiomeNoise(biomeNoisePos, 0.0050f, vec2(1835.32f, 3019.39f), 0.12f);
    return noise;
}

__device__ float getSingleCaveBiomeNoise(vec3 pos, float noiseScale, vec3 offset, float smoothstepThreshold)
{
    return smoothstep(-smoothstepThreshold, smoothstepThreshold, simplex(pos * noiseScale + offset));
}

__device__ CaveBiomeNoise getCaveBiomeNoise(const vec3 worldBlockPos, float maxHeight)
{
    const vec3 noiseOffset = fbm3From3<3>(worldBlockPos * 0.0470f) * vec3(30.f, 24.f, 30.f);
    const vec3 caveBiomeNoisePos = (worldBlockPos + noiseOffset) * vec3(overallCaveBiomeScale, 1.f, overallCaveBiomeScale);

    const vec2 noisePos2d = vec2(caveBiomeNoisePos.x, caveBiomeNoisePos.z) * 0.2000f;

    float caveNoiseTopHeight = SEA_LEVEL + 0.15f * (maxHeight - SEA_LEVEL);

    float noneToShallowStart = caveNoiseTopHeight - 16.f + 26.f * fbm<3>(noisePos2d);
    float noneToShallowEnd = noneToShallowStart - 5.f + 3.f * fbm<3>(noisePos2d + vec2(3821.34f, 4920.32f));

    float shallowToDeepStart = caveNoiseTopHeight - 72.f + 18.f * fbm<3>(noisePos2d + vec2(-4921.34f, 8402.13f));
    float shallowToDeepEnd = shallowToDeepStart - 10.f + 7.f * fbm<3>(noisePos2d + vec2(9411.32f, -3921.34f));

    CaveBiomeNoise noise;
    noise.none = smoothstep(noneToShallowEnd, noneToShallowStart, caveBiomeNoisePos.y);
    noise.shallow = smoothstep(shallowToDeepEnd, shallowToDeepStart, caveBiomeNoisePos.y);
    noise.warped = getSingleCaveBiomeNoise(caveBiomeNoisePos, 0.0030f, vec3(5821.32f, 4920.12f, 7931.59f), 0.05f);
    noise.rocky = getSingleCaveBiomeNoise(caveBiomeNoisePos, 0.0022f, vec3(-9193.23f, -6813.39f, -2171.23), 0.05f);
    return noise;
}

__device__ void applySingleBiomeNoise(float& totalWeight, const BiomeWeightType weight, const float noise)
{
    switch (weight)
    {
    case BiomeWeightType::W_POSITIVE:
        totalWeight *= noise;
        break;
    case BiomeWeightType::W_NEGATIVE:
        totalWeight *= 1.f - noise;
        break;
    }
}

__device__ float getBiomeWeight(Biome biome, const BiomeNoise& noise)
{
    const auto& biomeWeights = dev_biomeNoiseWeights[(int)biome];

    float totalWeight = 1.f;
#define applyNoise(type) applySingleBiomeNoise(totalWeight, biomeWeights.type, noise.type)
    applyNoise(ocean);
    applyNoise(beach);
    applyNoise(rocky);
    applyNoise(magic);
    applyNoise(temperature);
    applyNoise(moisture);
#undef applyNoise
    return totalWeight;
}

__device__ float getCaveBiomeWeight(CaveBiome biome, const CaveBiomeNoise& noise)
{
    const auto& caveBiomeWeights = dev_caveBiomeNoiseWeights[(int)biome];

    float totalWeight = 1.f;
#define applyNoise(type) applySingleBiomeNoise(totalWeight, caveBiomeWeights.type, noise.type)
    applyNoise(none);
    applyNoise(shallow);
    applyNoise(warped);
    applyNoise(rocky);
#undef applyNoise
    return totalWeight;
}

__device__ CaveBiome getCaveBiome(ivec3 worldBlockPos, float maxHeight, int seed)
{
    CaveBiomeNoise noise = getCaveBiomeNoise(worldBlockPos, maxHeight);

    auto rng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z, seed);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);
    for (int caveBiomeIdx = 0; caveBiomeIdx < numCaveBiomes; ++caveBiomeIdx)
    {
        CaveBiome caveBiome = (CaveBiome)caveBiomeIdx;
        float weight = getCaveBiomeWeight(caveBiome, noise);
        rand -= weight;
        if (rand <= 0.f)
        {
            return caveBiome;
        }
    }

    return CaveBiome::NONE;
}

#pragma endregion

__device__ float getHeight(Biome biome, vec2 pos)
{
    switch (biome)
    {
    case Biome::CORAL_REEF:
    {
        return 107.f + 16.f * fbm(pos * 0.0065f);
    }
    case Biome::ARCHIPELAGO:
    {
        float islandNoise = (fbm<4>(pos * 0.0055f) + 1.f) * 0.5f;
        islandNoise = powf(islandNoise, 2.4f);
        islandNoise = smoothstep(1.f, 0.f, islandNoise);
        float islandHeight = 22.f * islandNoise;

        float baseHeight = 107.f + 24.f * fbm(pos * 0.0060f);
        return baseHeight + islandHeight;
    }
    case Biome::WARM_OCEAN:
    {
        return 93.f + 18.f * fbm(pos * 0.0055f);
    }
    case Biome::ICEBERGS:
    {
        return 66.f + 18.f * fbm(pos * 0.0060f);
    }
    case Biome::COOL_OCEAN:
    {
        return 80.f + 22.f * fbm(pos * 0.0065f);
    }
    case Biome::ROCKY_BEACH:
    {
        return 134.f + 8.f * fbm(pos * 0.0070f);
    }
    case Biome::TROPICAL_BEACH:
    {
        return 129.5f + 6.f * fbm(pos * 0.0045f);
    }
    case Biome::BEACH:
    {
        return 132.f + 5.f * fbm(pos * 0.0055f);
    }
    case Biome::SAVANNA:
    {
        vec2 noiseOffsetPos = pos * 0.0040f;
        vec2 noiseOffset = fbm2From2<5>(noiseOffsetPos) * 100.f;

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
        vec2 noiseOffset = fbm2From2<5>(noiseOffsetPos) * 300.f;
        float riverNoise;
        worley((pos + noiseOffset) * 0.0030f, nullptr, &riverNoise);

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
        vec2 noiseOffset = simplex2From2(pos * 0.0080f) * 20.0f;
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
        vec2 noiseOffset = simplex2From2(pos * 0.0800f) * 3.0f;
        vec2 noisePos = (pos + noiseOffset) * 0.0300f;

        float worley1 = smoothstep(0.45f, 0.35f, worley(noisePos)) * 1.2f;
        float worley2 = smoothstep(0.45f, 0.35f, worley(noisePos * 1.4f + vec2(4292.12f, 9183.27f))) * 0.6f;
        float mountainsHeight = worley1 + worley2;
        mountainsHeight *= 54.f + 7.f * fbm<3>(noisePos * 1.7f);

        float hillsHeight = 16.f * simplex(pos * 0.0150f);

        return 128.f + hillsHeight + 9.f * fbm<3>(pos * 0.0070f) + mountainsHeight;
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
        float towersWorleyNoise;
        worley(pos * 0.0700f, &worleyColor, &towersWorleyNoise);
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

__device__ bool biomeBlockPreProcess(Block* blockPtr, Biome biome, ivec3 worldBlockPos, float height)
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
                *blockPtr = Block::QUARTZ;
                return true;
            }
        }

        return false;
    }
    }

    return false;
}

__device__ bool biomeBlockPostProcess(Block* blockPtr, Biome biome, ivec3 worldBlockPos, float height, bool isTopBlock)
{
    switch (biome)
    {
    case Biome::ARCHIPELAGO:
    {
        if (worldBlockPos.y < SEA_LEVEL || *blockPtr == Block::WATER)
        {
            return false;
        }

        float dirtHeight = SEA_LEVEL + 1.5f + 1.7f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0065f);
        if (worldBlockPos.y > dirtHeight)
        {
            *blockPtr = isTopBlock ? Block::GRASS_BLOCK : Block::DIRT;
            return true;
        }

        return false;
    }
    case Biome::TROPICAL_BEACH:
    {
        if (isTopBlock && *blockPtr != Block::SMOOTH_SAND && *blockPtr != Block::WATER)
        {
            *blockPtr = Block::SMOOTH_SAND;
            return true;
        }

        return false;
    }
    case Biome::BEACH:
    {
        if (isTopBlock && *blockPtr != Block::SAND && *blockPtr != Block::WATER)
        {
            *blockPtr = Block::SAND;
            return true;
        }

        return false;
    }
    case Biome::MESA:
    {
        if (worldBlockPos.y < 90.f || *blockPtr == Block::WATER)
        {
            return false;
        }

        vec2 pos2d = vec2(worldBlockPos.x, worldBlockPos.z);
        float terracottaStartHeight = 108.f + 12.f * fbm<3>(pos2d * 0.0040f);
        if (worldBlockPos.y < terracottaStartHeight)
        {
            return false;
        }

        if (*blockPtr == Block::CLAY && worldBlockPos.y < terracottaStartHeight + 20.f)
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

        *blockPtr = terracottaBlock;
        return true;
    }
    case Biome::FROZEN_WASTELAND:
    {
        if (*blockPtr != Block::WATER)
        {
            return false;
        }

        *blockPtr = Block::PACKED_ICE;
        return true;
    }
    case Biome::SHREKS_SWAMP:
    {
        if (worldBlockPos.y < 100.f)
        {
            return false;
        }

        if (*blockPtr == Block::DIRT || *blockPtr == Block::JUNGLE_GRASS_BLOCK)
        {
            float mudEnd = SEA_LEVEL + 0.8f + 1.1f * simplex(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0300f);
            if (worldBlockPos.y < mudEnd)
            {
                *blockPtr = Block::MUD;
                return true;
            }
        }

        return false;
    }
    case Biome::TIANZI_MOUNTAINS:
    {
        if (worldBlockPos.y < 90.f || *blockPtr == Block::WATER || *blockPtr == Block::DIRT || *blockPtr == Block::GRASS_BLOCK)
        {
            return false;
        }

        float sandstoneStartHeight = 112.f + 16.f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0200f);

        if (worldBlockPos.y < sandstoneStartHeight)
        {
            return false;
        }

        *blockPtr = Block::SMOOTH_SANDSTONE;
        return true;
    }
    case Biome::MOUNTAINS:
    {
        if (worldBlockPos.y < 190.f)
        {
            return false;
        }

        float snowStartHeight = 202.f + 5.f * fbm<3>(vec2(worldBlockPos.x, worldBlockPos.z) * 0.0500f);
        if (worldBlockPos.y < snowStartHeight)
        {
            return false;
        }

        *blockPtr = Block::SNOW;
        return true;
    }
    }

    return false;
}

__device__ bool caveBiomeBlockPostProcess(Block* blockPtr, CaveBiome caveBiome, ivec3 worldBlockPos, int caveBottomDepth, int caveTopDepth)
{
    if (caveBiome == CaveBiome::NONE)
    {
        return false;
    }

    bool isTopBlock = caveBottomDepth == 0;
    bool isBottomBlock = caveTopDepth == 0;

    switch (caveBiome)
    {
    case CaveBiome::CRYSTAL_CAVES:
    {
        if (*blockPtr != Block::STONE && *blockPtr != Block::DEEPSLATE && *blockPtr != Block::BLACKSTONE)
        {
            return false;
        }

        vec3 noisePos = vec3(worldBlockPos.x + worldBlockPos.y, worldBlockPos.z + 5819323, (worldBlockPos.x + worldBlockPos.z) * 2.0f) * 0.05f;
        float quartzNoise = simplex(noisePos);
        if (quartzNoise < -0.25f)
        {
            *blockPtr = Block::QUARTZ;
            return true;
        }

        if (*blockPtr == Block::BLACKSTONE)
        {
            return false;
        }

        float cobblestoneChance;
        Block cobblestoneBlock;
        if (*blockPtr == Block::STONE)
        {
            cobblestoneChance = 0.5f;
            cobblestoneBlock = Block::COBBLESTONE;
        }
        else
        {
            cobblestoneChance = 0.4f;
            cobblestoneBlock = Block::COBBLED_DEEPSLATE;
        }

        if (rand1From3(worldBlockPos) < cobblestoneChance)
        {
            *blockPtr = cobblestoneBlock;
            return true;
        }

        return false;
    }
    case CaveBiome::LUSH_CAVES:
    {
        if (*blockPtr != Block::STONE && *blockPtr != Block::DEEPSLATE && *blockPtr != Block::BLACKSTONE)
        {
            return false;
        }

        vec3 noisePos = vec3(worldBlockPos) * 0.025f;
        float threshold = 1.5f + 4.5f * simplex(noisePos);
        if (!isInRange((float)caveBottomDepth, 0.f, threshold) && !isInRange((float)caveTopDepth, 0.f, threshold))
        {
            return false;
        }

        noisePos.y += 192031.9821f;
        vec3 noiseOffset = fbm3From3<3>(noisePos * 0.4f) * 2.f;
        float clayNoise = worley(noisePos + noiseOffset);

        *blockPtr = clayNoise < 0.25f ? Block::CLAY : Block::MOSS;
        return true;
    }
    case CaveBiome::WARPED_FOREST:
    {
        if (!isTopBlock)
        {
            return false;
        }

        if (*blockPtr == Block::DEEPSLATE)
        {
            *blockPtr = Block::WARPED_DEEPSLATE;
            return true;
        }
        else if (*blockPtr == Block::BLACKSTONE)
        {
            *blockPtr = Block::WARPED_BLACKSTONE;
            return true;
        }

        return false;
    }
    case CaveBiome::AMBER_FOREST:
    {
        if (!isTopBlock)
        {
            return false;
        }

        if (*blockPtr == Block::DEEPSLATE)
        {
            *blockPtr = Block::AMBER_DEEPSLATE;
            return true;
        }
        else if (*blockPtr == Block::BLACKSTONE)
        {
            *blockPtr = Block::AMBER_BLACKSTONE;
            return true;
        }

        return false;
    }
    }
}

__constant__ BiomeBlocks dev_biomeBlocks[numBiomes];
__constant__ MaterialInfo dev_materialInfos[numMaterials];
__constant__ float dev_biomeMaterialWeights[numBiomes * numMaterials];

__constant__ ivec2 dev_dirVecs2d[8];

static std::array<std::vector<FeatureGen>, numBiomes> host_biomeFeatureGens;
static std::array<ivec2, numFeatures> host_featureHeightBounds;
static std::array<std::vector<CaveFeatureGen>, numCaveBiomes> host_caveBiomeFeatureGens;
static std::array<ivec2, numCaveFeatures> host_caveFeatureHeightBounds;
__constant__ ivec2 dev_featureHeightBounds[numFeatures];
__constant__ ivec2 dev_caveFeatureHeightBounds[numCaveFeatures];

static std::array<std::vector<DecoratorGen>, numBiomes> host_biomeDecoratorGens;
static std::array<std::vector<DecoratorGen>, numCaveBiomes> host_caveBiomeDecoratorGens;

void BiomeUtils::init()
{
#define biomeWeights(biome) host_biomeNoiseWeights[(int)Biome::biome]
#define caveBiomeWeights(caveBiome) host_caveBiomeNoiseWeights[(int)CaveBiome::caveBiome]
#define wI BiomeWeightType::W_IGNORE
#define wP BiomeWeightType::W_POSITIVE
#define wN BiomeWeightType::W_NEGATIVE

    BiomeWeights* host_biomeNoiseWeights = new BiomeWeights[numBiomes];

                                        // ocean, beach, rocky, magic, temperature, moisture
    biomeWeights(CORAL_REEF) =          { wP, wN, wP, wP, wI, wI };
    biomeWeights(ARCHIPELAGO) =         { wP, wN, wP, wN, wI, wI };
    biomeWeights(WARM_OCEAN) =          { wP, wN, wN, wI, wP, wI };
    biomeWeights(ICEBERGS) =            { wP, wN, wN, wP, wN, wI };
    biomeWeights(COOL_OCEAN) =          { wP, wN, wN, wN, wN, wI };

    biomeWeights(ROCKY_BEACH) =         { wP, wP, wP, wI, wI, wI };
    biomeWeights(TROPICAL_BEACH) =      { wP, wP, wN, wI, wP, wI };
    biomeWeights(BEACH) =               { wP, wP, wN, wI, wN, wI };

    biomeWeights(SAVANNA) =             { wN, wI, wP, wP, wP, wP };
    biomeWeights(MESA) =                { wN, wI, wP, wP, wP, wN };
    biomeWeights(FROZEN_WASTELAND) =    { wN, wI, wP, wP, wN, wP };
    biomeWeights(REDWOOD_FOREST) =      { wN, wI, wP, wP, wN, wN };
    biomeWeights(SHREKS_SWAMP) =        { wN, wI, wP, wN, wP, wP };
    biomeWeights(SPARSE_DESERT) =       { wN, wI, wP, wN, wP, wN };
    biomeWeights(LUSH_BIRCH_FOREST) =   { wN, wI, wP, wN, wN, wP };
    biomeWeights(TIANZI_MOUNTAINS) =    { wN, wI, wP, wN, wN, wN };

    biomeWeights(JUNGLE) =              { wN, wI, wN, wP, wP, wP };
    biomeWeights(RED_DESERT) =          { wN, wI, wN, wP, wP, wN };
    biomeWeights(PURPLE_MUSHROOMS) =    { wN, wI, wN, wP, wN, wP };
    biomeWeights(CRYSTALS) =            { wN, wI, wN, wP, wN, wN };
    biomeWeights(OASIS) =               { wN, wI, wN, wN, wP, wP };
    biomeWeights(DESERT) =              { wN, wI, wN, wN, wP, wN };
    biomeWeights(PLAINS) =              { wN, wI, wN, wN, wN, wP };
    biomeWeights(MOUNTAINS) =           { wN, wI, wN, wN, wN, wN };

    cudaMemcpyToSymbol(dev_biomeNoiseWeights, host_biomeNoiseWeights, numBiomes * sizeof(BiomeWeights));
    delete[] host_biomeNoiseWeights;

    CaveBiomeWeights* host_caveBiomeNoiseWeights = new CaveBiomeWeights[numCaveBiomes];

                                        // none, warped, rocky
    caveBiomeWeights(NONE) =            { wP, wI, wI, wI };
    
    caveBiomeWeights(CRYSTAL_CAVES) =   { wN, wP, wI, wP };
    caveBiomeWeights(LUSH_CAVES) =      { wN, wP, wI, wN };

    caveBiomeWeights(WARPED_FOREST) =   { wI, wN, wP, wI };
    caveBiomeWeights(AMBER_FOREST) =    { wI, wN, wN, wI };

    cudaMemcpyToSymbol(dev_caveBiomeNoiseWeights, host_caveBiomeNoiseWeights, numCaveBiomes * sizeof(CaveBiomeWeights));
    delete[] host_caveBiomeNoiseWeights;

#undef biomeWeights
#undef wI
#undef wP
#undef wN

    BiomeBlocks* host_biomeBlocks = new BiomeBlocks[numBiomes];

    host_biomeBlocks[(int)Biome::TROPICAL_BEACH].grassBlock = Block::JUNGLE_GRASS_BLOCK;

    host_biomeBlocks[(int)Biome::SAVANNA].grassBlock = Block::SAVANNA_GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::FROZEN_WASTELAND].grassBlock = Block::SNOWY_GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::REDWOOD_FOREST].grassBlock = Block::GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::SHREKS_SWAMP].grassBlock = Block::JUNGLE_GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::LUSH_BIRCH_FOREST].grassBlock = Block::GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::TIANZI_MOUNTAINS].grassBlock = Block::GRASS_BLOCK;

    host_biomeBlocks[(int)Biome::JUNGLE].grassBlock = Block::JUNGLE_GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::PURPLE_MUSHROOMS].grassBlock = Block::MYCELIUM;
    host_biomeBlocks[(int)Biome::OASIS].grassBlock = Block::JUNGLE_GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::PLAINS].grassBlock = Block::GRASS_BLOCK;
    host_biomeBlocks[(int)Biome::MOUNTAINS].grassBlock = Block::GRASS_BLOCK;

    cudaMemcpyToSymbol(dev_biomeBlocks, host_biomeBlocks, numBiomes * sizeof(BiomeBlocks));
    delete[] host_biomeBlocks;

#pragma region material infos

    MaterialInfo* host_materialInfos = new MaterialInfo[numMaterials];

#define setMaterialInfo(material, block, v1, v2, v3) host_materialInfos[(int)Material::material] = { Block::block, v1, v2, v3 }
#define setMaterialInfoSameBlock(material, v1, v2, v3) setMaterialInfo(material, material, v1, v2, v3)

    // material/block, thickness, noise amplitude, noise scale
    setMaterialInfoSameBlock(BLACKSTONE, 32.f, 32.f, 0.0030f);
    setMaterialInfoSameBlock(DEEPSLATE, 66.f, 20.f, 0.0045f);
    setMaterialInfoSameBlock(SLATE, 6.f, 24.f, 0.0062f);
    setMaterialInfoSameBlock(STONE, 40.f, 30.f, 0.0050f);
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

    setBiomeMaterialWeight(CORAL_REEF, DIRT, 0.0f);
    setBiomeMaterialWeight(CORAL_REEF, SAND, 0.7f);

    setBiomeMaterialWeight(ARCHIPELAGO, GRAVEL, 0.3f);
    setBiomeMaterialWeight(ARCHIPELAGO, DIRT, 0.0f);
    setBiomeMaterialWeight(ARCHIPELAGO, SAND, 0.8f);

    setBiomeMaterialWeight(WARM_OCEAN, DIRT, 0.0f);
    setBiomeMaterialWeight(WARM_OCEAN, SAND, 0.7f);

    setBiomeMaterialWeight(ICEBERGS, GRAVEL, 0.5f);
    setBiomeMaterialWeight(ICEBERGS, DIRT, 0.0f);

    setBiomeMaterialWeight(COOL_OCEAN, GRAVEL, 0.5f);
    setBiomeMaterialWeight(COOL_OCEAN, DIRT, 0.0f);

    setBiomeMaterialWeight(ROCKY_BEACH, DIRT, 0.0f);
    setBiomeMaterialWeight(ROCKY_BEACH, GRAVEL, 1.0f);

    setBiomeMaterialWeight(TROPICAL_BEACH, DIRT, 0.0f);
    setBiomeMaterialWeight(TROPICAL_BEACH, SMOOTH_SAND, 1.0f);

    setBiomeMaterialWeight(BEACH, DIRT, 0.0f);
    setBiomeMaterialWeight(BEACH, SAND, 1.0f);

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

#pragma region feature/decorator gens

    // for surface features, actual bounds = (pos.y + bounds[0], pos.y + bounds[1])
#define setFeatureHeightBounds(feature, yMin, yMax) host_featureHeightBounds[(int)Feature::feature] = ivec2(yMin, yMax)

    // feature, gridCellSize, gridCellPadding, chancePerGridCell, possibleTopLayers, canReplaceBlocks
    host_biomeFeatureGens[(int)Biome::ICEBERGS] = {
        { Feature::ICEBERG, 112, 6, 0.70f, {} }
    };

    host_biomeFeatureGens[(int)Biome::TROPICAL_BEACH] = {
        { Feature::PALM_TREE, 48, 3, 0.35f, { {Material::SMOOTH_SAND, 0.3f} } }
    };

    host_biomeFeatureGens[(int)Biome::SAVANNA] = {
        { Feature::ACACIA_TREE, 36, 4, 0.3f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::REDWOOD_FOREST] = {
        { Feature::REDWOOD_TREE, 10, 2, 0.75f, { {Material::DIRT, 0.5f} } }
    };

    host_biomeFeatureGens[(int)Biome::SHREKS_SWAMP] = {
        { Feature::CYPRESS_TREE, 18, 3, 0.6f, { {Material::DIRT, 0.5f}, {Material::MUD, 0.5f} } },
        { Feature::BIRCH_TREE, 16, 2, 0.15f, { {Material::DIRT, 0.4f} } }
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
        { Feature::LARGE_JUNGLE_TREE, 28, 3, 0.70f, { {Material::DIRT, 0.5f} } },
        { Feature::SMALL_JUNGLE_TREE, 10, 2, 0.82f, { {Material::DIRT, 0.5f} } },
        { Feature::TINY_JUNGLE_TREE, 6, 1, 0.28f, { {Material::DIRT, 0.5f} } }
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

    setFeatureHeightBounds(NONE, 0, 0);
    setFeatureHeightBounds(SPHERE, -6, 6);

    setFeatureHeightBounds(ICEBERG, 0, 110); // placed at sea level

    setFeatureHeightBounds(ACACIA_TREE, 0, 15);

    setFeatureHeightBounds(REDWOOD_TREE, -5, 75);

    setFeatureHeightBounds(CYPRESS_TREE, -3, 50);

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

    cudaMemcpyToSymbol(dev_featureHeightBounds, host_featureHeightBounds.data(), numFeatures * sizeof(ivec2));

    host_biomeDecoratorGens[(int)Biome::TROPICAL_BEACH] = {
        { Block::JUNGLE_GRASS, 0.1f, { Block::JUNGLE_GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::SAVANNA] = {
        { Block::SAVANNA_GRASS, 0.1f, { Block::SAVANNA_GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::REDWOOD_FOREST] = {
        { Block::GRASS, 0.2f, { Block::GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::SHREKS_SWAMP] = {
        { Block::JUNGLE_GRASS, 0.3f, { Block::JUNGLE_GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::LUSH_BIRCH_FOREST] = {
        { Block::GRASS, 0.3f, { Block::GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::JUNGLE] = {
        { Block::JUNGLE_GRASS, 0.4f, { Block::JUNGLE_GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::OASIS] = {
        { Block::JUNGLE_GRASS, 0.2f, { Block::JUNGLE_GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::PLAINS] = {
        { Block::GRASS, 0.2f, { Block::GRASS_BLOCK } }
    };

    host_biomeDecoratorGens[(int)Biome::MOUNTAINS] = {
        { Block::GRASS, 0.04f, { Block::GRASS_BLOCK } }
    };

#undef setFeatureHeightBounds
#pragma endregion

#pragma region cave feature/decorator gens

    // for cave features, actual bounds = (pos.y - bounds[0], pos.y + height + bounds[1])
#define setCaveFeatureHeightBounds(caveFeature, paddingBottom, paddingTop) host_caveFeatureHeightBounds[(int)CaveFeature::caveFeature] = ivec2(paddingBottom, paddingTop)

    // caveFeature, gridCellSize, gridCellPadding, chancePerGridCell
    host_caveBiomeFeatureGens[(int)CaveBiome::CRYSTAL_CAVES] = {
        CaveFeatureGen(CaveFeature::STORMLIGHT_SPHERE, 20, 3, 0.80f).setMinLayerHeight(4),
        CaveFeatureGen(CaveFeature::CRYSTAL_PILLAR, 28, 5, 0.60f).setMinLayerHeight(10).setNotReplaceBlocks().setGeneratesFromCeiling()
    };

    host_caveBiomeFeatureGens[(int)CaveBiome::LUSH_CAVES] = {
        CaveFeatureGen(CaveFeature::GLOWSTONE_CLUSTER, 24, 3, 0.60f).setMinLayerHeight(16).setNotReplaceBlocks().setGeneratesFromCeiling(),
        CaveFeatureGen(CaveFeature::CAVE_VINE, 4, 0, 0.40f).setMinLayerHeight(4).setNotReplaceBlocks().setGeneratesFromCeiling()
    };

    host_caveBiomeFeatureGens[(int)CaveBiome::WARPED_FOREST] = {
        CaveFeatureGen(CaveFeature::GLOWSTONE_CLUSTER, 16, 3, 0.80f).setMinLayerHeight(16).setNotReplaceBlocks().setGeneratesFromCeiling(),
        CaveFeatureGen(CaveFeature::WARPED_FUNGUS, 7, 1, 0.75f).setMinLayerHeight(6).setNotReplaceBlocks()
    };

    host_caveBiomeFeatureGens[(int)CaveBiome::AMBER_FOREST] = {
        CaveFeatureGen(CaveFeature::GLOWSTONE_CLUSTER, 18, 3, 0.75f).setMinLayerHeight(16).setNotReplaceBlocks().setGeneratesFromCeiling(),
        CaveFeatureGen(CaveFeature::AMBER_FUNGUS, 5, 1, 0.60f).setMinLayerHeight(9).setNotReplaceBlocks()
    };

    setCaveFeatureHeightBounds(NONE, 0, 0);
    setCaveFeatureHeightBounds(TEST_GLOWSTONE_PILLAR, -3, 3);
    setCaveFeatureHeightBounds(TEST_SHROOMLIGHT_PILLAR, -3, 3);

    setCaveFeatureHeightBounds(CAVE_VINE, 0, 0);

    setCaveFeatureHeightBounds(GLOWSTONE_CLUSTER, 0, 6);

    setCaveFeatureHeightBounds(STORMLIGHT_SPHERE, -7, 7);
    setCaveFeatureHeightBounds(CRYSTAL_PILLAR, -8, 8);

    setCaveFeatureHeightBounds(WARPED_FUNGUS, -2, 3);
    setCaveFeatureHeightBounds(AMBER_FUNGUS, -2, 5);

    cudaMemcpyToSymbol(dev_caveFeatureHeightBounds, host_caveFeatureHeightBounds.data(), numCaveFeatures * sizeof(ivec2));

    host_caveBiomeDecoratorGens[(int)CaveBiome::WARPED_FOREST] = {
        { Block::WARPED_MUSHROOM, 0.02f, { Block::WARPED_DEEPSLATE, Block::WARPED_BLACKSTONE } },
        { Block::WARPED_ROOTS, 0.06f, { Block::WARPED_DEEPSLATE, Block::WARPED_BLACKSTONE } },
        { Block::NETHER_SPROUTS, 0.04f, { Block::WARPED_DEEPSLATE, Block::WARPED_BLACKSTONE } }
    };

    host_caveBiomeDecoratorGens[(int)CaveBiome::AMBER_FOREST] = {
        { Block::INFECTED_MUSHROOM, 0.02f, { Block::AMBER_DEEPSLATE, Block::AMBER_BLACKSTONE } },
        { Block::AMBER_ROOTS, 0.06f, { Block::AMBER_DEEPSLATE, Block::AMBER_BLACKSTONE } }
    };

#undef setCaveFeatureHeightBounds
#pragma endregion
}