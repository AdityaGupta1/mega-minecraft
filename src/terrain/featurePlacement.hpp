#pragma once

#include "cuda/cudaUtils.hpp"
#include "biome.hpp"
#include "util/rng.hpp"

#define CONTROL_POINTS 5
#define SPLINE_SIZE 7

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

        vec3 endPoint = vec3(0, height, 0);
        vec3 ctrlPts[CONTROL_POINTS];
        const float lastCtrlPtIndex = CONTROL_POINTS - 1;
        ctrlPts[0] = vec3(0);
        for (int i = 1; i < CONTROL_POINTS; ++i)
        {
            vec3 offset = vec3(u11(rng), u11(rng), u11(rng)) * vec3(6, 2, 6);
            if (i == CONTROL_POINTS - 1)
            {
                offset *= 0.6f;
            }
            ctrlPts[i] = (endPoint * (i / lastCtrlPtIndex)) + offset;
        }

        const int lastSplineIndex = SPLINE_SIZE - 1;
        vec3 spline[SPLINE_SIZE];
        for (int i = 0; i < SPLINE_SIZE; ++i)
        {
            vec3 ctrlPtsCopy[CONTROL_POINTS];

            for (int j = 0; j < CONTROL_POINTS; ++j)
            {
                ctrlPtsCopy[j] = ctrlPts[j];
            }

            int points = CONTROL_POINTS;
            float t = float(i) / lastSplineIndex;
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

        for (int i = 0; i < SPLINE_SIZE; ++i)
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
                pos2 = pos1 + normalize(pos1 - spline[i - 1]) * (1.5f * u01(rng) + 2.5f);
            }

            vec3 vecLine = pos2 - pos1;

            vec3 pointPos = pos - pos1;
            float ratio = dot(pointPos, vecLine) / dot(vecLine, vecLine);

            vec3 pointLine = vecLine * ratio;
            vec3 vecPointPos = pointPos - pointLine;

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

                if (length(vecPointPos) < radius - 1.8f && ratio < 0.5f && universalScale < 1.4f)
                {
                    potentialBlock = Block::MUSHROOM_UNDERSIDE;
                }
                else
                {
                    potentialBlock = Block::MUSHROOM_CAP_PURPLE;
                }
            }

            if ((ratio >= 0 && ratio <= 1 && length(vecPointPos) <= radius) 
                || (i < lastSplineIndex && ratio < 0 && distance(pos, pos1) < radius) 
                || (i < (SPLINE_SIZE - 2) && ratio > 1 && distance(pos, pos2) < radius))
            {
                *block = potentialBlock;
                return true;
            }
        }

        return false;
    }
    }
}
