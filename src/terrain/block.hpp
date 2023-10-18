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

struct SideUv
{
    SideUv() = default;
    SideUv(glm::vec2 uv) : uv(uv) {}

    glm::vec2 uv;
    bool randRot{ false };
    bool randFlip{ false };
};

struct BlockUvs
{
    BlockUvs() = default;
    BlockUvs(glm::vec2 all) : top(all), side(all), bottom(all) {}
    BlockUvs(glm::vec2 side, glm::vec2 vert) : top(vert), side(side), bottom(vert) {}
    BlockUvs(glm::vec2 side, glm::vec2 top, glm::vec2 bottom) : top(top), side(side), bottom(bottom) {}

    SideUv side;
    SideUv top;
    SideUv bottom;

    void normalize()
    {
        top.uv *= 0.0625f;
        side.uv *= 0.0625f;
        bottom.uv *= 0.0625f;
    }

    BlockUvs& setRandomRotation()
    {
        side.randRot = top.randRot = bottom.randRot = true;
        return *this;
    }

    BlockUvs& setRandomRotation(bool side, bool top, bool bottom)
    {
        this->side.randRot = side;
        this->top.randRot = top;
        this->bottom.randRot = bottom;
        return *this;
    }

    BlockUvs& setRandomFlip()
    {
        side.randFlip = top.randFlip = bottom.randFlip = true;
        return *this;
    }

    BlockUvs& setRandomFlip(bool side, bool top, bool bottom)
    {
        this->side.randFlip = side;
        this->top.randFlip = top;
        this->bottom.randFlip = bottom;
        return *this;
    }
};

struct BlockData
{
    BlockUvs uvs;
    // will later include things like transparent, x-shaped, etc.
};

namespace BlockUtils
{
    void init();

    BlockData getBlockData(Block block);
}