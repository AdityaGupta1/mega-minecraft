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
    SNOWY_GRASS,
    MUSHROOM_STEM,
    MUSHROOM_UNDERSIDE,
    MUSHROOM_CAP_PURPLE,
    MARBLE,
    ANDESITE,
    CALCITE,
    BLACKSTONE,
    TUFF,
    DEEPSLATE,
    GRANITE,
    SLATE,
    SANDSTONE,
    CLAY,
    RED_SAND,
    RED_SANDSTONE,
    MUD,
    JUNGLE_GRASS,
    RAFFLESIA_PETAL,
    RAFFLESIA_CENTER,
    RAFFLESIA_SPIKES,
    RAFFLESIA_STEM,
    JUNGLE_LOG,
    JUNGLE_LEAVES
};

static constexpr int numBlocks = (int)Block::JUNGLE_LEAVES + 1;

struct SideUv
{
    SideUv() = default;
    SideUv(glm::ivec2 uv) : uv(uv) {}

    glm::ivec2 uv;
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

enum class TransparencyType : unsigned char
{
    OPAQUE_BLOCK,
    SEMI_TRANSPARENT, // e.g. leaves
    TRANSPARENT_BLOCK, // e.g. glass
    X_SHAPED // e.g. flowers, tall grass
};

struct BlockData
{
    BlockUvs uvs;
    TransparencyType transparency{ TransparencyType::OPAQUE_BLOCK };
};

namespace BlockUtils
{
    void init();

    BlockData getBlockData(Block block);
}