#pragma once

#include <glm/glm.hpp>
#include <string>

using namespace glm;

#define PI           3.14159265358979323846264338327f
#define PI_OVER_TWO  1.57079632679489661923132169163f
#define PI_OVER_FOUR 0.78539816339744830961566084581f

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

    std::string readFile(const std::string& filePath);
}