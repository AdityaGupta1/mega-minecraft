#pragma once

#include <glm/glm.hpp>

enum class Block : unsigned char
{
    AIR,
    STONE,
    DIRT,
    GRASS,
    SAND,
    GRAVEL,
    MYCELIUM,
    SNOW,
    SNOWY_GRASS
};

struct BlockUvs
{
    BlockUvs() = default;
    BlockUvs(glm::vec2 all) : top(all), side(all), bottom(all) {}
    BlockUvs(glm::vec2 side, glm::vec2 vert) : top(vert), side(side), bottom(vert) {}
    BlockUvs(glm::vec2 side, glm::vec2 top, glm::vec2 bottom) : top(top), side(side), bottom(bottom) {}

    glm::vec2 side;
    glm::vec2 top;
    glm::vec2 bottom;

    bool randRotSide{ false };
    bool randRotTop{ false };
    bool randRotBottom{ false };

    bool randFlipSide{ false };
    bool randFlipTop{ false };
    bool randFlipBottom{ false };

    void normalize()
    {
        top *= 0.0625f;
        side *= 0.0625f;
        bottom *= 0.0625f;
    }

    BlockUvs& setRandomRotation()
    {
        randRotSide = randRotTop = randRotBottom = true;
        return *this;
    }

    BlockUvs& setRandomRotation(bool side, bool top, bool bottom)
    {
        randRotSide = side;
        randRotTop = top;
        randRotBottom = bottom;
        return *this;
    }

    BlockUvs& setRandomFlip()
    {
        randFlipSide = randFlipTop = randFlipBottom = true;
        return *this;
    }

    BlockUvs& setRandomFlip(bool side, bool top, bool bottom)
    {
        randFlipSide = side;
        randFlipTop = top;
        randFlipBottom = bottom;
        return *this;
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