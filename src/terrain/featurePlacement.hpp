#pragma once

#include "cuda/cudaUtils.hpp"
#include "biome.hpp"

// block should not change if return value is false
__device__ bool placeFeature(FeaturePlacement featurePlacement, ivec3 worldBlockPos, Block* block)
{
    const ivec3& featurePos = featurePlacement.pos;

    switch (featurePlacement.feature)
    {
    case Feature::SPHERE:
        glm::vec3 diff = worldBlockPos - featurePos;

        if (glm::dot(diff, diff) > 25.f)
        {
            return false;
        }

        *block = Block::GRAVEL;
        return true;
    }
}
