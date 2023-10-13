#pragma once

#include <glm/glm.hpp>

enum class Block : unsigned char
{
    AIR,
    STONE,
    DIRT,
    GRASS
};

struct BlockUvs
{
    BlockUvs() = default;
    BlockUvs(glm::vec2 all) : top(all), side(all), bottom(all) {}
    BlockUvs(glm::vec2 side, glm::vec2 vert) : top(vert), side(side), bottom(vert) {}
    BlockUvs(glm::vec2 side, glm::vec2 top, glm::vec2 bottom) : top(top), side(side), bottom(bottom) {}

    glm::vec2 top;
    glm::vec2 side;
    glm::vec2 bottom;

    void normalize()
    {
        top *= 0.0625f;
        side *= 0.0625f;
        bottom *= 0.0625f;
    }
};

struct BlockData
{
    BlockUvs uvs;
};

namespace BlockUtils
{
    void init();

    BlockData getBlockData(Block block);
}