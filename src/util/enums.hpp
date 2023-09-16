#pragma once

#include <glm/glm.hpp>
#include <array>

using namespace glm;

//enum Direction
//{
//    NORTH, // +z
//    EAST, // +x
//    SOUTH, // -z
//    WEST, // -x
//    UP, // +y
//    DOWN // -y
//};

enum Direction2D
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
    const static std::array<ivec2, 8> dirVecs2D = {
        ivec2(0, 1),
        ivec2(1, 1),
        ivec2(1, 0),
        ivec2(1, -1),
        ivec2(0, -1),
        ivec2(-1, -1),
        ivec2(-1, 0),
    };
}