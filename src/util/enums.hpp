#pragma once

#include <glm/glm.hpp>
#include <array>

using namespace glm;

enum class Direction
{
    NORTH, // +z
    EAST, // +x
    SOUTH, // -z
    WEST, // -x
    UP, // +y
    DOWN // -y
};

enum class Direction2d
{
    NORTH, // +z
    NORTHEAST,
    EAST, // +x
    SOUTHEAST,
    SOUTH, // -z
    SOUTHWEST,
    WEST, // -x
    NORTHWEST
};

namespace DirectionEnums
{
    const static std::array<ivec2, 8> dirVecs2d = {
        ivec2(0, 1),   // 0
        ivec2(1, 1),   // 1
        ivec2(1, 0),   // 2
        ivec2(1, -1),  // 3
        ivec2(0, -1),  // 4
        ivec2(-1, -1), // 5
        ivec2(-1, 0),  // 6
        ivec2(-1, 1),  // 7
    };

    const static std::array<ivec3, 6> dirVecs = {
        ivec3(0, 0, 1), // forward
        ivec3(1, 0, 0), // right
        ivec3(0, 0, -1), // back
        ivec3(-1, 0, 0), // left
        ivec3(0, 1, 0), // up
        ivec3(0, -1, 0) // down
    };
}