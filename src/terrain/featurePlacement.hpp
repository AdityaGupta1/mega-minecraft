#pragma once

#include "cuda/cudaUtils.hpp"
#include "biome.hpp"
#include "util/rng.hpp"
#include "util/utils.hpp"

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

__device__ void calculateLineParams(const vec3& pos, const vec3& linePos1, const vec3& linePos2, float* ratio, float* distFromLine)
{
    vec3 vecLine = linePos2 - linePos1;

    vec3 pointPos = pos - linePos1;
    *ratio = dot(pointPos, vecLine) / dot(vecLine, vecLine);

    vec3 pointLine = vecLine * (*ratio);
    *distFromLine = distance(pointPos, pointLine);
}

#pragma endregion

// block should not change if return value is false
__device__ bool placeFeature(FeaturePlacement featurePlacement, ivec3 worldBlockPos, Block* block)
{
    const ivec3& featurePos = featurePlacement.pos;
    vec3 pos = worldBlockPos - featurePos;

    auto rng = makeSeededRandomEngine(featurePos.x, featurePos.y, featurePos.z, 8);
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u11(-1, 1);

    switch (featurePlacement.feature)
    {
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
        float universalScale = 1.f + u01(rng) * 1.2f;
        pos *= universalScale;

        float height = 25.f + u01(rng) * 30.f;

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
            vec3 offset = vec3(u11(rng), u11(rng), u11(rng)) * vec3(6, 2, 6);
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
                pos2 = pos1 + normalize(pos1 - spline[i - 1]) * (3.f + u01(rng) * 1.5f);
            }

            float ratio;
            float distFromLine;
            calculateLineParams(pos, pos1, pos2, &ratio, &distFromLine);

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
                radius = (7.f * u01(rng) + 12.f) * mix(0.8f, 1.2f, (height - 33.f) / 40.f);

                if (distFromLine < radius - 1.8f && ratio < 0.5f && universalScale < 1.4f)
                {
                    potentialBlock = Block::MUSHROOM_UNDERSIDE;
                }
                else
                {
                    potentialBlock = Block::MUSHROOM_CAP_PURPLE;
                }
            }

            if ((ratio >= 0 && ratio <= 1 && distFromLine <= radius) // actually in the line
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

        const float petalStartAngle = u01(rng) * TWO_PI;
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
    case Feature::SMALL_JUNGLE_TREE:
        float height = 8.f + 4.f * u01(rng);

        if (pos.y < height && ivec2(floor(vec2(pos.x, pos.z))) == ivec2(0))
        {
            *block = Block::JUNGLE_LOG;
            return true;
        }

        float leavesY = pos.y - (height - 1.f);
        float leavesRadiusMultiplier = 0.8f + 1.2f * u01(rng);
        if (leavesY > 0.f && leavesY < 3.f)
        {
            float leavesRadius = mix(3.f, 1.8f, leavesY / 3.f) * leavesRadiusMultiplier;
            if (length(vec2(pos.x, pos.z)) < leavesRadius)
            {
                *block = Block::JUNGLE_LEAVES;
                return true;
            }
        }

        return false;
    }
}
