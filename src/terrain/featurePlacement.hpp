#pragma once

#include "cuda/cudaUtils.hpp"
#include "biome.hpp"
#include "util/rng.hpp"
#include "util/utils.hpp"
#include <glm/gtx/component_wise.hpp>
#include "biomeFuncs.hpp"
#include <glm/gtx/vector_angle.hpp>

#pragma region utility functions

__device__ int manhattanDistance(ivec3 a, ivec3 b)
{
    return compAdd(abs(a - b));
}

template<class T>
__device__ bool isInRange(T v, T min, T max)
{
    return v >= min && v <= max;
}

template<class T>
__device__ bool isPosInRange(T pos, T corner1, T corner2)
{
    T minPos = min(corner1, corner2);
    T maxPos = max(corner1, corner2);
    return pos.x >= minPos.x && pos.x <= maxPos.x 
        && pos.y >= minPos.y && pos.y <= maxPos.y
        && pos.z >= minPos.z && pos.z <= maxPos.z;
}

__device__ float saturate(float v)
{
    return clamp(v, 0.f, 1.f);
}

#pragma endregion

#pragma region SDFs

__device__ float sdSphere(vec3 p, float s)
{
    return length(p) - s;
}

__device__ float sdCappedCylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(vec2(p.x, p.z)), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
}

__device__ float opSubtraction(float d1, float d2) 
{ 
    return max(d1, -d2); 
}

__device__ float opOnion(float sdf, float thickness)
{
    return abs(sdf) - thickness;
}

#pragma endregion

#pragma region spline functions

template<int numCtrlPts, int splineSize>
__device__ void deCasteljau(vec3* ctrlPts, vec3* spline)
{
    for (int i = 0; i < splineSize; ++i)
    {
        vec3 ctrlPtsCopy[numCtrlPts];

        for (int j = 0; j < numCtrlPts; ++j)
        {
            ctrlPtsCopy[j] = ctrlPts[j];
        }

        int points = numCtrlPts;
        float t = float(i) / (splineSize - 1);
        while (points > 1)
        {
            for (int j = 0; j < points - 1; ++j)
            {
                ctrlPtsCopy[j] = mix(ctrlPtsCopy[j], ctrlPtsCopy[j + 1], t);
            }

            --points;
        }

        spline[i] = ctrlPtsCopy[0];
    }
}

__device__ bool calculateLineParams(const vec3& pos, const vec3& linePos1, const vec3& linePos2, float* ratio, float* distFromLine)
{
    vec3 vecLine = linePos2 - linePos1;

    vec3 pointPos = pos - linePos1;
    *ratio = dot(pointPos, vecLine) / dot(vecLine, vecLine);

    vec3 pointLine = vecLine * (*ratio);
    *distFromLine = distance(pointPos, pointLine);

    return *ratio >= 0 && *ratio <= 1;
}

#pragma endregion

#pragma region feature-specific functions

__device__ bool jungleLeaves(vec3 pos, float maxHeight, float minRadius, float maxRadius, float rand)
{
    float leavesRadiusMultiplier = 0.8f + 0.4f * rand;
    if (pos.y > 0 && pos.y < maxHeight)
    {
        float leavesRadius = mix(maxRadius, minRadius, pos.y / maxHeight) * leavesRadiusMultiplier;
        return length(vec2(pos.x, pos.z)) < leavesRadius;
    }

    return false;
}

constexpr float crystalConeStart = 0.8f;
constexpr float crystalConeN = 1.f / (1.f - crystalConeStart);

__device__ float getCrystalRadius(float ratio)
{
    if (ratio < crystalConeStart)
    {
        return 0.8f + 0.25f * ratio;
    }
    else
    {
        return crystalConeN * (1.f - ratio);
    }
}

__device__ bool isInCrystal(vec3 pos, vec3 pos1, vec3 pos2, float radiusMultiplier)
{
    float ratio;
    float distanceFromLine;
    bool inLine = calculateLineParams(pos, pos1, pos2, &ratio, &distanceFromLine);

    if (!inLine)
    {
        return false;
    }

    float crystalRadius = getCrystalRadius(ratio) * radiusMultiplier;
    constexpr float p = PI / 6.f;
    const vec3 line = pos2 - pos1;
    const vec3 pointPos = pos - (pos1 + ratio * line);
    const float posAngle = length(pointPos) == 0.f ? 0.f : (angle(normalize(pointPos), normalize(cross(line, vec3(1, 0, 0)))) + TWO_PI);
    crystalRadius *= cosf(p) / cosf(p - fmod(posAngle, 2.f * p));
    return distanceFromLine < crystalRadius;
}

#pragma endregion

// block should not change if return value is false
__device__ bool placeFeature(FeaturePlacement featurePlacement, ivec3 worldBlockPos, Block* block)
{
    const ivec3& featurePos = featurePlacement.pos;
    ivec3 floorPos = worldBlockPos - featurePos;
    vec3 pos = floorPos;

    auto featureRng = makeSeededRandomEngine(featurePos.x, featurePos.y, featurePos.z, 8);
    auto blockRng = makeSeededRandomEngine(worldBlockPos.x, worldBlockPos.y, worldBlockPos.z, 9);
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u11(-1, 1);

    switch (featurePlacement.feature)
    {
    case Feature::NONE:
    {
        return false;
    }
    case Feature::SPHERE:
    {
        glm::vec3 diff = worldBlockPos - featurePos;

        if (glm::dot(diff, diff) > 25.f)
        {
            return false;
        }

        *block = Block::GRAVEL;
        return true;
    }
    case Feature::PURPLE_MUSHROOM:
    {
        float universalScale = 1.f + u01(featureRng) * 1.2f;
        pos *= universalScale;

        float height = 25.f + u01(featureRng) * 30.f;

        if (pos.y < -1 || pos.y > height + 12 || (length(vec2(pos.x, pos.z)) > 8 && (pos.y < height - 12 || length(pos - vec3(0, height, 0)) > 35)))
        {
            return false;
        }

        constexpr int numCtrlPts = 5;
        constexpr int splineSize = 7;

        vec3 endPoint = vec3(0, height, 0);
        vec3 ctrlPts[numCtrlPts];
        constexpr float lastCtrlPtIndex = numCtrlPts - 1.f;
        ctrlPts[0] = vec3(0);
#pragma unroll
        for (int i = 1; i < numCtrlPts; ++i)
        {
            vec3 offset = vec3(u11(featureRng), u11(featureRng), u11(featureRng)) * vec3(6, 2, 6);
            if (i == numCtrlPts - 1)
            {
                offset *= 0.6f;
            }
            ctrlPts[i] = (endPoint * (i / lastCtrlPtIndex)) + offset;
        }

        constexpr int lastSplineIndex = splineSize - 1;
        vec3 spline[splineSize];
        deCasteljau<numCtrlPts, splineSize>(ctrlPts, spline);

        for (int i = 0; i < splineSize; ++i)
        {
            vec3 pos1 = spline[i];
            vec3 pos2;

            if (i < lastSplineIndex)
            {
                pos2 = spline[i + 1];

                if (pos.y < pos1.y - 3 || pos.y > pos2.y + 3)
                {
                    continue;
                }
            }
            else
            {
                pos2 = pos1 + normalize(pos1 - spline[i - 1]) * (3.f + u01(featureRng) * 1.5f);
            }

            float ratio;
            float distFromLine;
            bool inRatio = calculateLineParams(pos, pos1, pos2, &ratio, &distFromLine);

            float radius;
            Block potentialBlock;
            if (i < lastSplineIndex)
            {
                float t = (i + clamp(ratio, 0.f, 1.f)) / lastSplineIndex;
                float x = t - 0.5f;
                radius = (4.f * x * x + 1.5f) * 1.2f;
                potentialBlock = Block::MUSHROOM_STEM;
            }
            else
            {
                radius = (7.f * u01(featureRng) + 12.f) * mix(0.8f, 1.2f, (height - 33.f) / 40.f);

                if (distFromLine < radius - 1.8f && ratio < 0.5f && universalScale < 1.4f)
                {
                    potentialBlock = Block::MUSHROOM_UNDERSIDE;
                }
                else
                {
                    potentialBlock = Block::MUSHROOM_CAP_PURPLE;
                }
            }

            if ((inRatio && distFromLine <= radius) // actually in the line
                || (i < lastSplineIndex && ratio < 0 && distance(pos, pos1) < radius) // start cap
                || (i < (splineSize - 2) && ratio > 1 && distance(pos, pos2) < radius)) // end cap
            {
                *block = potentialBlock;
                return true;
            }
        }

        return false;
    }
    case Feature::RAFFLESIA:
    {
        if (pos.y > 10.f || length(pos) > 15.f)
        {
            return false;
        }

        pos *= 0.8f;

        vec3 centerSdfPos = pos;
        centerSdfPos.y -= 1.f;
        centerSdfPos.y *= 1.4f;

        if (sdSphere(centerSdfPos, 1.f) < 0)
        {
            *block = Block::RAFFLESIA_SPIKES;
            return true;
        }

        float centerSdf = opOnion(sdSphere(centerSdfPos - vec3(0, 1, 0), 2.0f), 0.8f);
        float centerHoleSdf = sdSphere(centerSdfPos - vec3(0, 1.8f, 0), 1.8f);
        centerSdf = opSubtraction(centerSdf, centerHoleSdf);

        if (centerSdf < 0.f)
        {
            *block = centerSdfPos.y > 1.f ? Block::RAFFLESIA_CENTER : Block::RAFFLESIA_STEM;
            return true;
        }

        const float petalStartAngle = u01(featureRng) * TWO_PI;
        for (int i = 0; i < 5; ++i)
        {
            const float petalAngle = petalStartAngle + (i * TWO_PI * 0.2f);

            float sinTheta, cosTheta;
            sincosf(-petalAngle, &sinTheta, &cosTheta);
            vec3 petalPos = vec3(pos.x * cosTheta + pos.z * sinTheta, pos.y - 3.2f, -pos.x * sinTheta + pos.z * cosTheta);
            petalPos.y -= (i % 2) * 0.53f;
            petalPos.y += clamp((abs(petalPos.x - 3.f) - 1.5f) / 1.5f, 0.f, 1.f) * 1.3f;
            petalPos.x -= 3.8f;
            petalPos.z *= 1.2f;

            if (sdCappedCylinder(petalPos, 2.5f, 0.5f) < 0)
            {
                *block = Block::RAFFLESIA_PETAL;
                return true;
            }
        }

        return false;
    }
    case Feature::LARGE_JUNGLE_TREE:
    {
        float height = 18.f + 10.f * u01(featureRng);
        if (pos.y > height + 6.f || length(vec2(pos.x, pos.z)) > 15.f)
        {
            return false;
        }

        ivec2 trunkPos = ivec2(floor(vec2(pos.x, pos.z)));
        if (isInRange(pos.y, 0.f, height) && trunkPos.x >= 0 && trunkPos.x <= 1 && trunkPos.y >= 0 && trunkPos.y <= 1)
        {
            *block = Block::JUNGLE_WOOD;
            return true;
        }

        pos -= vec3(0.5f, 0, 0.5f);

        vec3 leavesPos = pos;
        leavesPos.y -= (height - 2.f);
        if (jungleLeaves(leavesPos, 4.f, 4.f, 7.f, u01(featureRng)))
        {
            *block = u01(blockRng) < 0.5f ? Block::JUNGLE_LEAVES_FRUITS : Block::JUNGLE_LEAVES_PLAIN;
            return true;
        }

        float numBranches = 0.5f + 2.5f * u01(featureRng);
        float branchHeight = height;
        for (int i = 0; i < numBranches; ++i)
        {
            branchHeight -= (8.f + u01(featureRng) * 3.f) * (height / 30.f);
            float branchAngle = TWO_PI * u01(featureRng);

            vec3 branchStart = vec3(0, branchHeight, 0);

            vec3 branchEnd = vec3(0);
            sincosf(-branchAngle, &branchEnd.z, &branchEnd.x);
            branchEnd = ((3.f + 1.5f * u01(featureRng)) * branchEnd) + branchStart;
            branchEnd.y += 1.f + 1.5f * u01(featureRng);

            float ratio;
            float distFromLine;
            bool inRatio = calculateLineParams(pos, branchStart, branchEnd, &ratio, &distFromLine);

            float branchRadius = 1.2f - (0.4f * ratio);
            if (inRatio && distFromLine < branchRadius)
            {
                *block = Block::JUNGLE_WOOD;
                return true;
            }

            leavesPos = pos - branchEnd + vec3(0, 0.2f, 0);
            if (jungleLeaves(leavesPos, 2.f, 2.5f, 3.5f, u01(featureRng)))
            {
                *block = u01(blockRng) < 0.25f ? Block::JUNGLE_LEAVES_FRUITS : Block::JUNGLE_LEAVES_PLAIN;
                return true;
            }
        }

        return false;
    }
    case Feature::SMALL_JUNGLE_TREE:
    {
        float height = 8.f + 4.f * u01(featureRng);
        float maxDist = pos.y < height - 2.f ? 2.f : 8.f;
        if (pos.y > height + 4.f || length(vec2(pos.x, pos.z)) > maxDist)
        {
            return false;
        }

        if (isInRange(pos.y, 0.f, height) && ivec2(floor(vec2(pos.x, pos.z))) == ivec2(0))
        {
            *block = Block::JUNGLE_WOOD;
            return true;
        }

        vec3 leavesPos = pos - vec3(0, height - 1.f, 0);
        if (jungleLeaves(leavesPos, 3.f, 2.f, 4.f, u01(featureRng)))
        {
            *block = u01(blockRng) < 0.25f ? Block::JUNGLE_LEAVES_FRUITS : Block::JUNGLE_LEAVES_PLAIN;
            return true;
        }

        return false;
    }
    case Feature::TINY_JUNGLE_TREE:
    {
        if (compAdd(floorPos) > 8)
        {
            return false;
        }

        int height = (int)(0.5f + 2.5f * u01(featureRng));
        if (floorPos.x == 0 && isInRange(floorPos.y, 0, height) && floorPos.z == 0)
        {
            *block = Block::JUNGLE_WOOD;
            return true;
        }

        if (manhattanDistance(floorPos, vec3(0, height, 0)) == 1)
        {
            *block = Block::JUNGLE_LEAVES_PLAIN;
            return true;
        }

        return false;
    }
    case Feature::CACTUS:
    {
        if (abs(floorPos.x) > 5 || abs(floorPos.z) > 5)
        {
            return false;
        }

        int height = (int)(7.5f + u01(featureRng) * 6.0f);

        if (pos.y > height + 2.f)
        {
            return false;
        }

        if (floorPos.x == 0 && isInRange(floorPos.y, 0, height) && floorPos.z == 0)
        {
            *block = Block::CACTUS;
            return true;
        }

        for (int armIdx = 0; armIdx < 4; ++armIdx)
        {
            if (u01(featureRng) >= 0.35f)
            {
                continue;
            }

            int armStartHeight = (int)(4.f + u01(featureRng) * (height - 10));
            int armLength = (int)(2.f + u01(featureRng) * 1.f);
            int armHeight = (int)(3.f + u01(featureRng) * 3.f);
            armHeight = min(height - armStartHeight - 1, armHeight);

            ivec3 armPos1 = ivec3(0, armStartHeight, 0);
            const ivec2 armDirection = dev_dirVecs2d[armIdx * 2];
            ivec3 armPos2 = armPos1 + (ivec3(armDirection.x, 0, armDirection.y) * armLength);
            ivec3 armPos3 = armPos2 + ivec3(0, armHeight, 0);

            if (isPosInRange(floorPos, armPos1, armPos2) || isPosInRange(floorPos, armPos2, armPos3))
            {
                *block = Block::CACTUS;
                return true;
            }
        }

        return false;
    }
    case Feature::PALM_TREE:
    {
        if (floorPos.y < -2 || floorPos.y > 28 || abs(floorPos.x) + abs(floorPos.z) > 24)
        {
            return false;
        }

        constexpr int numCtrlPts = 4;
        constexpr int splineSize = 5;

        vec3 minPos = vec3(0);
        vec3 maxPos = vec3(0);

        vec3 ctrlPts[numCtrlPts];
        vec3 currentPoint = vec3(0);
        ctrlPts[0] = currentPoint;
        for (int i = 1; i < numCtrlPts; ++i)
        {
            float randomWalkScale = 1.f + ((float)i / numCtrlPts) * 5.f;
            currentPoint += vec3(randomWalkScale * u11(featureRng), 3.f + 5.f * u01(featureRng), randomWalkScale * u11(featureRng));
            ctrlPts[i] = currentPoint;

            minPos = min(minPos, currentPoint);
            maxPos = max(maxPos, currentPoint);
        }

        if (!isPosInRange(pos, minPos - vec3(7, 1, 7), maxPos + vec3(7, 6, 7)))
        {
            return false;
        }

        vec3 spline[splineSize];
        deCasteljau<numCtrlPts, splineSize>(ctrlPts, spline);

        ivec3 trunkTop = ivec3(floor(spline[splineSize - 1]));
        ivec3 leavesPos = floorPos - trunkTop;
        float leavesDistance = length(vec2(leavesPos.x, leavesPos.z));
        leavesDistance *= 0.6f + (0.3f * saturate((20 - trunkTop.y) * 0.05f)) + (0.3f * u01(featureRng));
        if (isInRange(leavesPos.y, -1, 0) && leavesDistance < 3.9f
            && (leavesPos.x == 0 || leavesPos.z == 0 || abs(leavesPos.x) == abs(leavesPos.z)))
        {
            int leavesHeight = leavesDistance > 3.f ? -1 : 0;
            if (leavesPos.y == leavesHeight)
            {
                *block = Block::PALM_LEAVES;
                return true;
            }
        }

        for (int i = 0; i < splineSize - 1; ++i)
        {
            vec3 pos1 = spline[i];
            vec3 pos2 = spline[i + 1];
            vec3 padding = normalize(pos2 - pos1) * 0.5f;
            if (i > 0)
            {
                pos1 -= padding;
            }
            if (i + 1 < splineSize - 1)
            {
                pos2 += padding;
            }

            float ratio;
            float distFromLine;
            bool inLine = calculateLineParams(pos + vec3(0.5f), pos1, pos2, &ratio, &distFromLine);
            if (!inLine || distFromLine > 2.f)
            {
                continue;
            }

            const ivec3 actualPos = ivec3(floor(mix(pos1, pos2, ratio)));
            if (floorPos == actualPos)
            {
                *block = Block::PALM_WOOD;
                return true;
            }
        }

        return false;
    }
    case Feature::CRYSTAL:
    {
        pos += vec3(0, 2, 0);
        pos *= 0.6f + 0.4f * u01(featureRng);

        if (max(abs(pos.x), abs(pos.z)) > 15)
        {
            return false;
        }

        float crystalRand = u01(featureRng) * 3.f;

        vec3 crystalEndPos = vec3(12.f * u11(featureRng), 18.f + 8.f * u01(featureRng), 12.f * u11(featureRng));
        if (pos.y > crystalEndPos.y + 2.f)
        {
            return false;
        }

        if (!isInCrystal(pos, vec3(0), crystalEndPos, 4.f + 1.2f * u01(featureRng)))
        {
            pos *= 0.8f;

            bool isInSmallCrystal = false;

            int numSmallCrystals = (int)(4.f + 2.f * u01(featureRng));
            float smallCrystalAngle = u01(featureRng) * TWO_PI;
            for (int i = 0; i < numSmallCrystals; ++i)
            {
                smallCrystalAngle += PI_OVER_TWO + PI * u01(featureRng);
                vec3 smallCrystalStartPos = vec3(0);
                sincosf(smallCrystalAngle, &smallCrystalStartPos.z, &smallCrystalStartPos.x);
                vec3 smallCrystalEndPos = smallCrystalStartPos;
                smallCrystalStartPos *= 3.f;
                smallCrystalEndPos *= 6.f + 3.f * u01(featureRng);
                smallCrystalEndPos.y = 7.f + 5.f * u01(featureRng);

                if (isInCrystal(pos, vec3(0), smallCrystalEndPos, 1.5f + 1.5f * u01(featureRng)))
                {
                    isInSmallCrystal = true;
                    break;
                }
            }

            if (!isInSmallCrystal)
            {
                return false;
            }
        }

        Block crystalBlock;
        if (crystalRand < 1.f)
        {
            crystalBlock = Block::MAGENTA_CRYSTAL;
        }
        else if (crystalRand < 2.f)
        {
            crystalBlock = Block::CYAN_CRYSTAL;
        }
        else
        {
            crystalBlock = Block::GREEN_CRYSTAL;
        }

        *block = crystalBlock;
        return true;

        return false;
    }
    }

    printf("placeFeature() reached an unreachable section");
    return false;
}
