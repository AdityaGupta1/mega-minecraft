#pragma once

#include <glm/glm.hpp>

using namespace glm;

namespace Utils
{
    struct PosHash
    {
        size_t operator()(const glm::ivec2& k)const
        {
            return std::hash<int>()(k.x) ^ std::hash<int>()(k.y);
        }
    };

    ivec2 worldPosToChunkPos(vec3 worldPos);
}